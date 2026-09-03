import os
import re
import argparse
from datetime import datetime, timedelta
from typing import Optional

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from pyproj import Proj
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from igra_gen.utils import io
from igra_gen.models.precond import EDMPrecond
from igra_gen.generating.factory import sampler_factory


ERA5_ROOT = os.environ.get("ERA5_ROOT", "")
HYDRA_CFG = os.environ.get("ATMOSPHERIC_PRIOR_HYDRA_CONFIG", "")
DEFAULT_CHECKPOINT = os.environ.get("ATMOSPHERIC_PRIOR_CHECKPOINT", "")
GOES_LSF_ROOT = os.environ.get("GOES_LSF_ROOT", "")
HOURLY_AIRTEMP2KM_ROOT = os.environ.get("HOURLY_AIRTEMP2KM_ROOT", "")
SAMPLES_DIR = os.environ.get("ATMOSPHERIC_SAMPLES_DIR", "samples")
NUM_CHANNELS = 13

AIRTEMP2KM_SCALE = 0.00341802
AIRTEMP2KM_OFFSET = 149.0
AIRTEMP2KM_NODATA = 65535
AIRTEMP2KM_PHYS_MIN_K = 220.0
AIRTEMP2KM_PHYS_MAX_K = 330.0


def timestep_to_datetime(timestep: int) -> datetime:
    return datetime(2020, 1, 1, 0, 0, 0) + timedelta(hours=6 * timestep)


def _parse_goes_start_time(path: str):
    name = os.path.basename(path)
    # Example token: s20200010001184
    match = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", name)
    if match is None:
        return None
    year = int(match.group(1))
    day_of_year = int(match.group(2))
    hour = int(match.group(3))
    minute = int(match.group(4))
    second = int(match.group(5))
    return datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour, minutes=minute, seconds=second)


def _nearest_lsf_file(goes_lsf_root: str, target_dt: datetime):
    year = target_dt.year
    doy = target_dt.strftime("%j")
    hh = f"{target_dt.hour:02d}"
    candidate_dirs = [
        os.path.join(goes_lsf_root, "ABI-L2-LSTC", str(year), doy, hh),
        os.path.join(goes_lsf_root, "ABI-L2-LST", str(year), doy, hh),
    ]
    files = []
    for d in candidate_dirs:
        if os.path.isdir(d):
            files.extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".nc")])
    if not files:
        raise FileNotFoundError(f"No GOES LSF files under {candidate_dirs}")
    files.sort()
    scored = []
    for p in files:
        dt = _parse_goes_start_time(p)
        if dt is None:
            scored.append((10**9, p))
        else:
            scored.append((abs((dt - target_dt).total_seconds()), p))
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _infer_lst_kelvin(ds: xr.Dataset):
    var_candidates = ["LST", "land_surface_temperature", "Temp"]
    lst_name = None
    for v in var_candidates:
        if v in ds.variables:
            lst_name = v
            break
    if lst_name is None:
        for v in ds.data_vars:
            if any(k in v.lower() for k in ["lst", "temp", "surface"]):
                lst_name = v
                break
    if lst_name is None:
        raise ValueError("Could not find LST variable in GOES file.")

    lst = ds[lst_name].values.astype(np.float32)
    if np.nanmax(lst) < 150:
        # Celsius-like; convert to Kelvin to match ERA5 2m_temperature units.
        lst = lst + 273.15
    return lst


def hourly_airtemp2km_path(root: str, target_dt: datetime) -> str:
    """Return the HourlyAirTemp2kmUSA file matching a 2020 UTC timestamp."""
    doy = target_dt.strftime("%j")
    name = f"{target_dt.year}{doy}{target_dt.hour:02d}.nc"
    return os.path.join(root, "y2020avg", f"doy{doy}", name)


def load_hourly_airtemp2km(
    root: str,
    target_dt: datetime,
    mask_policy: str = "physical",
):
    """Load HourlyAirTemp2kmUSA air temperature in Kelvin plus a validity mask.

    The raw variable is an encoded uint-like field named ``at``. The official
    mask is ``at != 65535``; the physical mask additionally removes rare values
    outside a conservative near-surface air-temperature range.
    """
    if mask_policy not in {"official", "physical"}:
        raise ValueError(f"Unknown HourlyAirTemp2kmUSA mask_policy={mask_policy}")

    path = hourly_airtemp2km_path(root, target_dt)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"HourlyAirTemp2kmUSA file not found for {target_dt.isoformat()}: {path}"
        )

    ds = xr.open_dataset(path, mask_and_scale=False, engine="h5netcdf")
    try:
        if "at" not in ds.variables:
            raise ValueError(f"HourlyAirTemp2kmUSA file has no 'at' variable: {path}")
        raw = np.asarray(ds["at"].values)
        airtemp_k = raw.astype(np.float32) * AIRTEMP2KM_SCALE + AIRTEMP2KM_OFFSET
        official_valid = raw != AIRTEMP2KM_NODATA
        physical_valid = (
            official_valid
            & np.isfinite(airtemp_k)
            & (airtemp_k > AIRTEMP2KM_PHYS_MIN_K)
            & (airtemp_k < AIRTEMP2KM_PHYS_MAX_K)
        )
        valid = physical_valid if mask_policy == "physical" else official_valid
        airtemp_k = airtemp_k.astype(np.float32)
        airtemp_k[~official_valid] = np.nan
        return path, airtemp_k, valid
    finally:
        ds.close()


def _goes_xy_to_latlon(ds: xr.Dataset):
    x = ds["x"].values.astype(np.float64)
    y = ds["y"].values.astype(np.float64)
    proj = ds["goes_imager_projection"]

    sat_height = float(proj.perspective_point_height)
    sat_lon = float(proj.longitude_of_projection_origin)
    sat_sweep = str(proj.sweep_angle_axis)

    proj_str = f"+proj=geos +h={sat_height} +lon_0={sat_lon} +sweep={sat_sweep} +ellps=GRS80"
    p = Proj(proj_str)
    xx, yy = np.meshgrid(x * sat_height, y * sat_height)
    lons, lats = p(xx, yy, inverse=True)
    return lats.astype(np.float32), lons.astype(np.float32)


def _interpolate_uncertainty_to_points(uncertainty_map: np.ndarray, locations: np.ndarray) -> np.ndarray:
    """Interpolate a 128x256 uncertainty map to GOES point locations.

    This mirrors the DJ IGRAOperator.H latitude/longitude normalization used
    inside the posterior likelihood. The uncertainty map is rolled in longitude
    before grid_sample, matching the observation operator.
    """
    if uncertainty_map.shape != (128, 256):
        raise ValueError(f"Expected uncertainty map shape (128, 256), got {uncertainty_map.shape}")
    field = torch.as_tensor(uncertainty_map, dtype=torch.float32)
    field = torch.roll(field, shifts=128, dims=1).view(1, 1, 128, 256)
    pts = torch.as_tensor(locations, dtype=torch.float32)
    norm_lat = 2 * (pts[:, 0] - (-90.0)) / 180.0 - 1
    norm_lon = 2 * (pts[:, 1] + 180.0) / 360.0 - 1
    grid = torch.stack((norm_lon, norm_lat), dim=-1).view(1, -1, 1, 2)
    vals = F.grid_sample(field, grid, mode="bilinear", align_corners=True, padding_mode="reflection")
    return vals.squeeze(0).squeeze(0).squeeze(-1).numpy()


def _load_uncertainty_map(path: str, temp_idx: int = 0) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 4 or arr.shape[1:] != (NUM_CHANNELS, 128, 256):
        raise ValueError(f"Expected unconditional samples with shape (ens, {NUM_CHANNELS}, 128, 256), got {arr.shape} from {path}")
    return np.asarray(arr[:, temp_idx].std(axis=0), dtype=np.float32)


def _select_by_uncertainty(
    valid_idx: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    scores: np.ndarray,
    max_points: int,
    method: str,
    spread_cell_deg: float,
) -> np.ndarray:
    if max_points is None or max_points <= 0 or valid_idx.size <= max_points:
        return valid_idx

    order = np.argsort(scores)[::-1]
    if method == "uncertainty_topN":
        return valid_idx[order[:max_points]]

    if method != "uncertainty_spread":
        raise ValueError(f"Unknown uncertainty selection method: {method}")

    # Greedy grid-cell thinning: take the highest-uncertainty point per
    # lat/lon cell first, then fill any remaining budget by uncertainty rank.
    lat_flat = lats.ravel()[valid_idx]
    lon_flat = lons.ravel()[valid_idx]
    selected_positions = []
    seen_cells = set()
    for pos in order:
        cell = (int(np.floor(lat_flat[pos] / spread_cell_deg)), int(np.floor(lon_flat[pos] / spread_cell_deg)))
        if cell in seen_cells:
            continue
        seen_cells.add(cell)
        selected_positions.append(pos)
        if len(selected_positions) >= max_points:
            break

    if len(selected_positions) < max_points:
        selected_set = set(selected_positions)
        for pos in order:
            if pos in selected_set:
                continue
            selected_positions.append(pos)
            if len(selected_positions) >= max_points:
                break

    return valid_idx[np.asarray(selected_positions, dtype=np.int64)]


def _normalized_latlon_features(lat_flat: np.ndarray, lon_flat: np.ndarray) -> np.ndarray:
    lat_min, lat_max = float(np.nanmin(lat_flat)), float(np.nanmax(lat_flat))
    lon_min, lon_max = float(np.nanmin(lon_flat)), float(np.nanmax(lon_flat))
    lat_scale = max(lat_max - lat_min, 1e-6)
    lon_scale = max(lon_max - lon_min, 1e-6)
    lat_n = (lat_flat - lat_min) / lat_scale
    lon_n = (lon_flat - lon_min) / lon_scale
    return np.stack([lat_n, lon_n], axis=1).astype(np.float32)


def _select_spatial_uniform_grid(
    valid_idx: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    """Select a simple spatially spread baseline from the valid GOES pool.

    This is deliberately not uncertainty-aware. It makes a roughly square
    lat/lon grid over the same-timestep valid GOES points and picks the point
    closest to each occupied cell center. If the occupied-cell representatives
    are slightly fewer or more than the budget, a deterministic seeded fill/trim
    keeps the final count exact.
    """
    if max_points is None or max_points <= 0 or valid_idx.size <= max_points:
        return valid_idx

    rng = np.random.default_rng(seed)
    lat_flat = lats.ravel()[valid_idx]
    lon_flat = lons.ravel()[valid_idx]

    lat_span = max(float(np.nanmax(lat_flat) - np.nanmin(lat_flat)), 1e-6)
    lon_span = max(float(np.nanmax(lon_flat) - np.nanmin(lon_flat)), 1e-6)
    n_lat = max(1, int(round(np.sqrt(max_points * lat_span / lon_span))))
    n_lon = max(1, int(np.ceil(max_points / n_lat)))

    lat_edges = np.linspace(float(np.nanmin(lat_flat)), float(np.nanmax(lat_flat)), n_lat + 1)
    lon_edges = np.linspace(float(np.nanmin(lon_flat)), float(np.nanmax(lon_flat)), n_lon + 1)
    lat_bin = np.clip(np.searchsorted(lat_edges, lat_flat, side="right") - 1, 0, n_lat - 1)
    lon_bin = np.clip(np.searchsorted(lon_edges, lon_flat, side="right") - 1, 0, n_lon - 1)

    reps = []
    for iy in range(n_lat):
        for ix in range(n_lon):
            members = np.where((lat_bin == iy) & (lon_bin == ix))[0]
            if members.size == 0:
                continue
            lat_c = 0.5 * (lat_edges[iy] + lat_edges[iy + 1])
            lon_c = 0.5 * (lon_edges[ix] + lon_edges[ix + 1])
            dist2 = (lat_flat[members] - lat_c) ** 2 + (lon_flat[members] - lon_c) ** 2
            reps.append(members[int(np.argmin(dist2))])

    selected_positions = list(dict.fromkeys(reps))
    if len(selected_positions) > max_points:
        keep = rng.choice(np.asarray(selected_positions), size=max_points, replace=False)
        selected_positions = list(np.sort(keep))

    if len(selected_positions) < max_points:
        selected_set = set(selected_positions)
        remaining = np.setdiff1d(np.arange(valid_idx.size), np.fromiter(selected_set, dtype=np.int64), assume_unique=False)
        need = max_points - len(selected_positions)
        if remaining.size > 0:
            fill = rng.choice(remaining, size=min(need, remaining.size), replace=False)
            selected_positions.extend(fill.tolist())

    return valid_idx[np.asarray(selected_positions[:max_points], dtype=np.int64)]


def _select_kmeans(
    valid_idx: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    scores: Optional[np.ndarray],
    max_points: int,
    seed: int,
    weighted: bool,
) -> np.ndarray:
    if max_points is None or max_points <= 0 or valid_idx.size <= max_points:
        return valid_idx

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    lat_flat = lats.ravel()[valid_idx]
    lon_flat = lons.ravel()[valid_idx]
    features = _normalized_latlon_features(lat_flat, lon_flat)

    fit_limit = min(valid_idx.size, 200_000)
    if valid_idx.size > fit_limit:
        if weighted and scores is not None:
            w = np.asarray(scores, dtype=np.float64)
            w = w - np.nanmin(w)
            w = w + max(np.nanmean(w), 1e-8)
            p = w / np.nansum(w)
            fit_pos = rng.choice(np.arange(valid_idx.size), size=fit_limit, replace=False, p=p)
        else:
            fit_pos = rng.choice(np.arange(valid_idx.size), size=fit_limit, replace=False)
    else:
        fit_pos = np.arange(valid_idx.size)

    fit_features = features[fit_pos]
    sample_weight = None
    if weighted and scores is not None:
        fit_scores = np.asarray(scores, dtype=np.float64)[fit_pos]
        span = max(float(np.nanmax(fit_scores) - np.nanmin(fit_scores)), 1e-8)
        sample_weight = 1.0 + 9.0 * (fit_scores - np.nanmin(fit_scores)) / span

    kmeans = MiniBatchKMeans(
        n_clusters=max_points,
        random_state=seed,
        batch_size=max(4096, max_points * 20),
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
    )
    kmeans.fit(fit_features, sample_weight=sample_weight)

    nbrs = NearestNeighbors(n_neighbors=min(16, valid_idx.size)).fit(features)
    _, neigh = nbrs.kneighbors(kmeans.cluster_centers_)
    selected_positions = []
    selected_set = set()
    for row in neigh:
        for pos in row:
            pos = int(pos)
            if pos not in selected_set:
                selected_positions.append(pos)
                selected_set.add(pos)
                break

    if len(selected_positions) < max_points:
        if weighted and scores is not None:
            fill_order = np.argsort(scores)[::-1]
        else:
            fill_order = rng.permutation(valid_idx.size)
        for pos in fill_order:
            pos = int(pos)
            if pos in selected_set:
                continue
            selected_positions.append(pos)
            selected_set.add(pos)
            if len(selected_positions) >= max_points:
                break

    return valid_idx[np.asarray(selected_positions[:max_points], dtype=np.int64)]


def load_lsf_valid_pool(
    goes_lsf_root: str,
    target_dt: datetime,
    era5_mean_2m: float,
    era5_std_2m: float,
    observation_product: str = "raw_goes_lst",
    hourly_airtemp2km_root: str = HOURLY_AIRTEMP2KM_ROOT,
    airtemp_mask_policy: str = "physical",
):
    """Load the same-timestep valid GOES pixel pool before budget selection.

    ``raw_goes_lst`` keeps the original behavior: GOES LST values are used as
    the 2m-temperature observation. ``hourly_airtemp2kmusa`` keeps GOES valid
    pixel locations but replaces the observation values with HourlyAirTemp2kmUSA
    near-surface air temperature on the matching ABI CONUS grid.
    """
    if observation_product not in {"raw_goes_lst", "hourly_airtemp2kmusa"}:
        raise ValueError(f"Unknown observation_product={observation_product}")

    lsf_file = _nearest_lsf_file(goes_lsf_root, target_dt)
    ds = xr.open_dataset(lsf_file, engine="h5netcdf")
    try:
        lst_k = _infer_lst_kelvin(ds)
        lats, lons = _goes_xy_to_latlon(ds)
        dqf = ds["DQF"].values.astype(np.float32) if "DQF" in ds.variables else None
        observation_file = lsf_file
        observation_k = lst_k

        valid = np.isfinite(lst_k) & np.isfinite(lats) & np.isfinite(lons)
        valid = valid & (lst_k > 150.0) & (lst_k < 350.0)
        if dqf is not None:
            valid = valid & (dqf == 0)

        if observation_product == "hourly_airtemp2kmusa":
            observation_file, airtemp_k, airtemp_valid = load_hourly_airtemp2km(
                root=hourly_airtemp2km_root,
                target_dt=target_dt,
                mask_policy=airtemp_mask_policy,
            )
            if airtemp_k.shape != lst_k.shape:
                raise ValueError(
                    "HourlyAirTemp2kmUSA grid shape does not match GOES LST grid: "
                    f"{airtemp_k.shape} vs {lst_k.shape}"
                )
            observation_k = airtemp_k
            valid = valid & airtemp_valid & np.isfinite(observation_k)

        idx = np.where(valid.ravel())[0]
        if idx.size == 0:
            raise ValueError(f"No valid GOES pixels after filtering for {observation_product}.")

        lat_flat = lats.ravel()[idx]
        lon_flat = lons.ravel()[idx]
        obs_flat = observation_k.ravel()[idx]
        obs_norm = (obs_flat - era5_mean_2m) / era5_std_2m
        locations = np.stack([lat_flat, lon_flat], axis=1).astype(np.float32)
        values = obs_norm.astype(np.float32)
        return {
            "lsf_file": lsf_file,
            "observation_file": observation_file,
            "observation_product": observation_product,
            "idx": idx.astype(np.int64),
            "locations": locations,
            "values": values,
            "observation_k": observation_k,
            "lats": lats,
            "lons": lons,
            "lst_k": lst_k,
            "dqf": dqf,
        }
    finally:
        ds.close()


def load_lsf_measurements(
    goes_lsf_root: str,
    target_dt: datetime,
    max_points: Optional[int],
    seed: int,
    era5_mean_2m: float,
    era5_std_2m: float,
    selection_method: str = "random",
    uncertainty_path: Optional[str] = None,
    spread_cell_deg: float = 0.5,
    selected_indices_path: Optional[str] = None,
    observation_product: str = "raw_goes_lst",
    hourly_airtemp2km_root: str = HOURLY_AIRTEMP2KM_ROOT,
    airtemp_mask_policy: str = "physical",
):
    pool = load_lsf_valid_pool(
        goes_lsf_root=goes_lsf_root,
        target_dt=target_dt,
        era5_mean_2m=era5_mean_2m,
        era5_std_2m=era5_std_2m,
        observation_product=observation_product,
        hourly_airtemp2km_root=hourly_airtemp2km_root,
        airtemp_mask_policy=airtemp_mask_policy,
    )
    lsf_file = pool["lsf_file"]
    idx = pool["idx"]
    lats = pool["lats"]
    lons = pool["lons"]
    observation_k = pool["observation_k"]
    lat_flat = pool["locations"][:, 0]
    lon_flat = pool["locations"][:, 1]
    obs_flat = observation_k.ravel()[idx]

    if selected_indices_path:
        selected = np.load(selected_indices_path)
        if "selected_flat_indices" not in selected:
            raise ValueError(f"{selected_indices_path} must contain selected_flat_indices")
        selected_idx = np.asarray(selected["selected_flat_indices"], dtype=np.int64)
        valid_sorted = np.sort(np.asarray(idx, dtype=np.int64))
        is_valid = np.isin(selected_idx, valid_sorted, assume_unique=False)
        if not bool(np.all(is_valid)):
            bad = selected_idx[~is_valid][:10].tolist()
            raise ValueError(f"{selected_indices_path} contains indices that are not valid for this GOES timestep: {bad}")
        idx = selected_idx
        lat_flat = lats.ravel()[idx]
        lon_flat = lons.ravel()[idx]
        obs_flat = observation_k.ravel()[idx]
    elif max_points is not None and max_points > 0 and idx.size > max_points:
        if selection_method == "random":
            rng = np.random.default_rng(seed)
            keep = rng.choice(np.arange(idx.size), size=max_points, replace=False)
            idx = idx[keep]
            lat_flat = lat_flat[keep]
            lon_flat = lon_flat[keep]
            obs_flat = obs_flat[keep]
        elif selection_method == "spatial_uniform_grid":
            idx = _select_spatial_uniform_grid(
                valid_idx=idx,
                lats=lats,
                lons=lons,
                max_points=max_points,
                seed=seed,
            )
            lat_flat = lats.ravel()[idx]
            lon_flat = lons.ravel()[idx]
            obs_flat = observation_k.ravel()[idx]
        elif selection_method == "spatial_kmeans":
            idx = _select_kmeans(
                valid_idx=idx,
                lats=lats,
                lons=lons,
                scores=None,
                max_points=max_points,
                seed=seed,
                weighted=False,
            )
            lat_flat = lats.ravel()[idx]
            lon_flat = lons.ravel()[idx]
            obs_flat = observation_k.ravel()[idx]
        elif selection_method in {"uncertainty_topN", "uncertainty_spread"}:
            if uncertainty_path is None:
                raise ValueError(f"{selection_method} requires --uncertainty_path")
            locations_all = np.stack([lat_flat, lon_flat], axis=1).astype(np.float32)
            uncertainty_map = _load_uncertainty_map(uncertainty_path)
            scores = _interpolate_uncertainty_to_points(uncertainty_map, locations_all)
            idx = _select_by_uncertainty(
                valid_idx=idx,
                lats=lats,
                lons=lons,
                scores=scores,
                max_points=max_points,
                method=selection_method,
                spread_cell_deg=spread_cell_deg,
            )
            lat_flat = lats.ravel()[idx]
            lon_flat = lons.ravel()[idx]
            obs_flat = observation_k.ravel()[idx]
        elif selection_method == "uncertainty_weighted_kmeans":
            if uncertainty_path is None:
                raise ValueError(f"{selection_method} requires --uncertainty_path")
            locations_all = np.stack([lat_flat, lon_flat], axis=1).astype(np.float32)
            uncertainty_map = _load_uncertainty_map(uncertainty_path)
            scores = _interpolate_uncertainty_to_points(uncertainty_map, locations_all)
            idx = _select_kmeans(
                valid_idx=idx,
                lats=lats,
                lons=lons,
                scores=scores,
                max_points=max_points,
                seed=seed,
                weighted=True,
            )
            lat_flat = lats.ravel()[idx]
            lon_flat = lons.ravel()[idx]
            obs_flat = observation_k.ravel()[idx]
        else:
            raise ValueError(f"Unknown selection_method={selection_method}")

    obs_norm = (obs_flat - era5_mean_2m) / era5_std_2m

    locations = np.stack([lat_flat, lon_flat], axis=1).astype(np.float32)
    values = obs_norm.astype(np.float32)
    return lsf_file, locations, values


def sample_with_lsf(
    timestep=0,
    ens=1,
    seed=17,
    num_steps=50,
    sigma_min=0.005,
    sigma_max=80,
    rho=7,
    S_churn=0.0,
    S_min=0.01,
    S_max=50.0,
    S_noise=1.003,
    max_points=None,
    selection_method="random",
    uncertainty_path=None,
    spread_cell_deg=0.5,
    selected_indices_path=None,
    observation_product="raw_goes_lst",
    hourly_airtemp2km_root=HOURLY_AIRTEMP2KM_ROOT,
    airtemp_mask_policy="physical",
    goes_lsf_root=GOES_LSF_ROOT,
    checkpoint=DEFAULT_CHECKPOINT,
    output=None,
):
    if max_points is None or max_points <= 0:
        point_mode = "all_valid"
    elif selection_method == "uncertainty_spread":
        point_mode = f"{selection_method}_{max_points}_cell{spread_cell_deg:g}deg"
    else:
        point_mode = f"{selection_method}_{max_points}"
    io.log0(
        f"Starting LSF-conditioned sampling | timestep={timestep} ens={ens} "
        f"point_mode={point_mode} max_points={max_points} "
        f"observation_product={observation_product}"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target_dt = timestep_to_datetime(timestep)
    means = np.load(os.path.join(ERA5_ROOT, "normalize_mean.npz"))
    stds = np.load(os.path.join(ERA5_ROOT, "normalize_std.npz"))
    mean_2m = float(np.asarray(means["2m_temperature"]).reshape(-1)[0])
    std_2m = float(np.asarray(stds["2m_temperature"]).reshape(-1)[0])

    # Read GOES before instantiating the ERA5/DJ dataset. In this environment,
    # importing/opening HDF5 through h5py first can make netCDF4/xarray fail on
    # GOES files with "NetCDF: HDF error".
    lsf_file, locations, values = load_lsf_measurements(
        goes_lsf_root=goes_lsf_root,
        target_dt=target_dt,
        max_points=max_points,
        seed=seed + timestep,
        era5_mean_2m=mean_2m,
        era5_std_2m=std_2m,
        selection_method=selection_method,
        uncertainty_path=uncertainty_path,
        spread_cell_deg=spread_cell_deg,
        selected_indices_path=selected_indices_path,
        observation_product=observation_product,
        hourly_airtemp2km_root=hourly_airtemp2km_root,
        airtemp_mask_policy=airtemp_mask_policy,
    )
    io.log0(
        f"Using GOES LSF file: {lsf_file} | target_dt={target_dt} | "
        f"point_mode={point_mode} requested_max_points={max_points} kept_points={len(values)} "
        f"observation_product={observation_product}"
    )

    conf = OmegaConf.load(HYDRA_CFG)
    conf.data.dataset.root = ERA5_ROOT
    conf.data.dataset.split = "test"
    dataset: Dataset = instantiate(conf.data.dataset, _convert_="object")

    net = EDMPrecond(
        model=conf.model,
        img_resolution=(128, 256),
        img_channels=NUM_CHANNELS,
        sigma_data=1,
        sigma_max=80,
        sigma_min=0.005,
        condition_channels=1,
    ).to(device).eval()

    io.log0(f"Loading checkpoint: {checkpoint}")
    chkpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in chkpt["ema"].items()}
    has_model_prefix = any(k.startswith("model.") for k in state_dict)
    if has_model_prefix:
        net.load_state_dict(state_dict)
    else:
        net.model.load_state_dict(state_dict)

    condition, _ = dataset.__getitem__(timestep)
    condition = condition.float()[None, :].to(device)
    in_shape = (1, NUM_CHANNELS, 128, 256)

    model_vars = dataset.variables[:-1]  # 13 output channels for Haiwen native checkpoint-037129
    if len(model_vars) != NUM_CHANNELS:
        raise ValueError(f"Expected {NUM_CHANNELS} model variables, got {len(model_vars)}: {model_vars}")
    temp_idx = model_vars.index("2m_temperature")
    n_channels = len(model_vars)
    ch_locs = [np.empty((0, 2), dtype=np.float32) for _ in range(n_channels)]
    ch_vals = [np.empty((0,), dtype=np.float32) for _ in range(n_channels)]
    ch_locs[temp_idx] = locations
    ch_vals[temp_idx] = values

    query_locations = [ch_locs]
    true_values = [ch_vals]
    measurement = [query_locations * ens, true_values * ens]

    sample_fn = sampler_factory(
        mode="edm_pos_sample",
        net=net,
        conditioning_type="igra",
        in_shape=(16, 32),
        target_shape=(128, 256),
    )

    outs = []
    for i in range(ens):
        generator = torch.Generator(device=device).manual_seed(seed + i)
        sample = sample_fn(
            measurement,
            generator,
            condition=condition,
            in_shape=in_shape,
            device=device,
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            S_churn=S_churn,
            S_min=S_min,
            S_max=S_max,
            S_noise=S_noise,
        ).detach().cpu().numpy()
        outs.append(sample.squeeze())
        io.log0(f"Completed sample {i+1}/{ens}")

    out_array = np.array(outs)
    if output is None:
        output = os.path.join(SAMPLES_DIR, f"lsf_t{timestep:04d}.npy")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.save(output, out_array)
    points_output = os.path.splitext(output)[0] + "_selected_points.npz"
    np.savez_compressed(
        points_output,
        locations=locations,
        values=values,
        lsf_file=np.asarray(lsf_file),
        timestep=np.asarray(timestep),
        target_datetime=np.asarray(target_dt.isoformat()),
        max_points=np.asarray(-1 if max_points is None else max_points),
        kept_points=np.asarray(len(values)),
        selection_method=np.asarray(selection_method),
        observation_product=np.asarray(observation_product),
        hourly_airtemp2km_root=np.asarray(hourly_airtemp2km_root),
        airtemp_mask_policy=np.asarray(airtemp_mask_policy),
        seed=np.asarray(seed),
        point_selection_seed=np.asarray(seed + timestep),
        spread_cell_deg=np.asarray(spread_cell_deg),
        uncertainty_path=np.asarray("" if uncertainty_path is None else uncertainty_path),
        selected_indices_path=np.asarray("" if selected_indices_path is None else selected_indices_path),
    )
    io.log0(f"Saved LSF-conditioned samples to {output}")
    io.log0(f"Saved selected GOES points to {points_output}")


def main():
    parser = argparse.ArgumentParser(description="GOES LSF-conditioned diffusion sampling")
    parser.add_argument("--timestep", type=int, default=0, help="ERA5 timestep (0 => 2020_0000)")
    parser.add_argument("--ens", type=int, default=1, help="Ensemble size")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--sigma_min", type=float, default=0.005)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--S_churn", type=float, default=0.0)
    parser.add_argument("--S_min", type=float, default=0.01)
    parser.add_argument("--S_max", type=float, default=50.0)
    parser.add_argument("--S_noise", type=float, default=1.003)
    parser.add_argument(
        "--max_points",
        type=int,
        default=None,
        help="Optional random GOES LST pixel budget. Omit or pass <=0 to use all valid GOES pixels.",
    )
    parser.add_argument(
        "--selection_method",
        type=str,
        default="random",
        choices=[
            "random",
            "spatial_uniform_grid",
            "spatial_kmeans",
            "uncertainty_topN",
            "uncertainty_spread",
            "uncertainty_weighted_kmeans",
        ],
        help="Point selection method used when --max_points is positive.",
    )
    parser.add_argument(
        "--uncertainty_path",
        type=str,
        default=None,
        help="Path to unconditional ensemble samples used by uncertainty-based selection.",
    )
    parser.add_argument(
        "--spread_cell_deg",
        type=float,
        default=0.5,
        help="Lat/lon cell size for uncertainty_spread grid-cell thinning.",
    )
    parser.add_argument(
        "--selected_indices_path",
        type=str,
        default=None,
        help="Optional .npz containing selected_flat_indices. When set, this overrides built-in point selection.",
    )
    parser.add_argument(
        "--observation_product",
        type=str,
        default="raw_goes_lst",
        choices=["raw_goes_lst", "hourly_airtemp2kmusa"],
        help=(
            "Observation values used for the 2m_temperature likelihood. "
            "raw_goes_lst preserves the old GOES LST behavior; "
            "hourly_airtemp2kmusa keeps valid GOES pixel locations but uses "
            "HourlyAirTemp2kmUSA air-temperature values."
        ),
    )
    parser.add_argument(
        "--hourly_airtemp2km_root",
        type=str,
        default=HOURLY_AIRTEMP2KM_ROOT,
        help="Root containing y2020avg/doyDDD/YYYYDDDHH.nc HourlyAirTemp2kmUSA files.",
    )
    parser.add_argument(
        "--airtemp_mask_policy",
        type=str,
        default="physical",
        choices=["official", "physical"],
        help="HourlyAirTemp2kmUSA mask policy when --observation_product=hourly_airtemp2kmusa.",
    )
    parser.add_argument("--goes_lsf_root", type=str, default=GOES_LSF_ROOT)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    sample_with_lsf(
        timestep=args.timestep,
        ens=args.ens,
        seed=args.seed,
        num_steps=args.num_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        S_churn=args.S_churn,
        S_min=args.S_min,
        S_max=args.S_max,
        S_noise=args.S_noise,
        max_points=args.max_points,
        selection_method=args.selection_method,
        uncertainty_path=args.uncertainty_path,
        spread_cell_deg=args.spread_cell_deg,
        selected_indices_path=args.selected_indices_path,
        observation_product=args.observation_product,
        hourly_airtemp2km_root=args.hourly_airtemp2km_root,
        airtemp_mask_policy=args.airtemp_mask_policy,
        goes_lsf_root=args.goes_lsf_root,
        checkpoint=args.checkpoint,
        output=args.output,
    )


if __name__ == "__main__":
    main()
