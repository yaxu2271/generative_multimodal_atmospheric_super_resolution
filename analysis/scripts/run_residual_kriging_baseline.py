#!/usr/bin/env python3
"""Training-period-background residual kriging for Reviewer 2 Major Comment 1.

The script implements an auditable non-diffusion baseline using the same R/A/S
observation files as the manuscript experiments.  Hyperparameters are selected
only on the 24 prespecified 2019 development cases and then frozen for 2020.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def configured_path(name: str) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else Path(f"__SET_{name}__")


RUN = configured_path("KRIGING_RUN_ROOT")
ERA5 = configured_path("ERA5_ROOT")
BACKGROUND = Path(os.environ.get(
    "KRIGING_BACKGROUND",
    RUN / "outputs/training_climatology_month_hour_regional.npz",
)).expanduser().resolve()
SELECTED = RUN / "outputs/2019_selected_hyperparameters.json"

IGRA_2019 = configured_path("IGRA_2019_PKL")
AIRCRAFT_2019 = configured_path("AIRCRAFT_2019_ROOT")
SURFACE_2019 = configured_path("SURFACE_2019_ROOT")
IGRA_2020 = configured_path("IGRA_2020_PKL")
AIRCRAFT_2020 = configured_path("AIRCRAFT_2020_ROOT")
SURFACE_2020 = configured_path("SURFACE_2020_ROOT")
TIMESTEPS_2020 = configured_path("EVALUATION_TIMESTEP_MANIFEST")
DIFFUSION_METRICS = configured_path("AREA_WEIGHTED_DIFFUSION_METRICS_CSV")

VARIABLES = [
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "geopotential_500", "geopotential_850", "u_component_of_wind_500",
    "u_component_of_wind_850", "v_component_of_wind_500",
    "v_component_of_wind_850", "temperature_500", "temperature_850",
    "specific_humidity_500", "specific_humidity_850",
]
AIRCRAFT_VARS = {
    "temperature_500", "temperature_850", "u_component_of_wind_500",
    "u_component_of_wind_850", "v_component_of_wind_500", "v_component_of_wind_850",
}
SURFACE_VARS = set(VARIABLES[:3])
ALLOWED_AIRCRAFT_SOURCES = np.asarray([0, 1, 5], dtype=np.int16)
TIMESTEPS_2019 = [0, 56, 124, 180, 236, 292, 360, 416, 480, 536, 604, 660,
                  724, 780, 848, 904, 972, 1028, 1092, 1148, 1216, 1272, 1336, 1392]

# Regional analysis buffer.  The scored domain is strict CONUS; the buffer
# retains nearby observations and background values for boundary interpolation.
BUFFER = {"lat_min": 14.0, "lat_max": 61.0, "lon_min": -141.0, "lon_max": -49.0}
CONUS = {"lat_min": 24.0, "lat_max": 50.0, "lon_min": -125.0, "lon_max": -66.0}
LENGTHS_KM = [
    float(x)
    for x in os.environ.get(
        "KRIGING_LENGTHS", "200,400,800,1200,1600,2400"
    ).split(",")
]
NEIGHBORS = [
    int(x) for x in os.environ.get("KRIGING_NEIGHBORS", "8,16,32,64").split(",")
]
NUGGETS = [
    float(x) for x in os.environ.get("KRIGING_NUGGETS", "0.05,0.20,0.50").split(",")
]
EARTH_KM_PER_DEG = 111.195
EARTH_RADIUS_KM = 6371.0088
PROJECTION_LAT = 38.0
DISTANCE_METRIC = os.environ.get(
    "KRIGING_DISTANCE_METRIC", "great_circle"
).strip().lower()
if DISTANCE_METRIC not in {"fixed_latitude_equirectangular", "great_circle"}:
    raise ValueError(
        "KRIGING_DISTANCE_METRIC must be 'fixed_latitude_equirectangular' "
        "or 'great_circle'"
    )
UNITS = {
    "2m_temperature": "K", "10m_u_component_of_wind": "m s-1",
    "10m_v_component_of_wind": "m s-1", "geopotential_500": "m2 s-2",
    "geopotential_850": "m2 s-2", "u_component_of_wind_500": "m s-1",
    "u_component_of_wind_850": "m s-1", "v_component_of_wind_500": "m s-1",
    "v_component_of_wind_850": "m s-1", "temperature_500": "K",
    "temperature_850": "K", "specific_humidity_500": "kg kg-1",
    "specific_humidity_850": "kg kg-1",
}


@lru_cache(maxsize=1)
def grids() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, slice, slice]:
    lat = np.load(ERA5 / "lat.npy").astype(np.float64)
    lon360 = np.load(ERA5 / "lon.npy").astype(np.float64)
    lon = ((lon360 + 180.0) % 360.0) - 180.0
    lat_idx = np.where((lat >= BUFFER["lat_min"]) & (lat <= BUFFER["lat_max"]))[0]
    lon_idx = np.where((lon >= BUFFER["lon_min"]) & (lon <= BUFFER["lon_max"]))[0]
    if not (np.all(np.diff(lat_idx) == 1) and np.all(np.diff(lon_idx) == 1)):
        raise RuntimeError("Regional buffer is not contiguous in the ERA5 grid")
    ls = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
    os_ = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
    return lat, lon, lat[ls], lon[os_], ls, os_


@lru_cache(maxsize=1)
def normalization() -> tuple[np.ndarray, np.ndarray]:
    mean_file = np.load(ERA5 / "normalize_mean.npz")
    std_file = np.load(ERA5 / "normalize_std.npz")
    mean = np.asarray([float(mean_file[v].reshape(-1)[0]) for v in VARIABLES])
    std = np.asarray([float(std_file[v].reshape(-1)[0]) for v in VARIABLES])
    return mean, std


def timestamp(year: int, timestep: int) -> datetime:
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(hours=6 * timestep)


def background_bin(month: int, hour: int) -> int:
    if hour not in (0, 12):
        raise ValueError(f"Expected 00 or 12 UTC, found {hour}")
    return (month - 1) * 2 + (hour // 12)


def build_background() -> None:
    """Build 1979--2018 monthly/00-12 UTC regional climatology."""
    _, _, rlat, rlon, lat_slice, lon_slice = grids()
    sums = np.zeros((24, len(VARIABLES), len(rlat), len(rlon)), dtype=np.float64)
    counts = np.zeros(24, dtype=np.int64)
    files = sorted((ERA5 / "train").glob("*.h5"))
    used = 0
    for index, path in enumerate(files, start=1):
        year = int(path.name[:4])
        step = int(path.stem.split("_")[1])
        dt = timestamp(year, step)
        if dt.hour not in (0, 12):
            continue
        b = background_bin(dt.month, dt.hour)
        with h5py.File(path, "r") as handle:
            for vi, variable in enumerate(VARIABLES):
                sums[b, vi] += np.asarray(handle["input"][variable][lat_slice, lon_slice], dtype=np.float64)
        counts[b] += 1
        used += 1
        if used == 1 or used % 1000 == 0:
            print(f"[background] used={used} scanned={index}/{len(files)}", flush=True)
    # Each bin contains every day of the corresponding month/hour across 40
    # years (not one sample per year).  Paired 00/12 UTC counts must match and
    # every month must contain at least 28 days x 40 years.
    if np.any(counts < 28 * 40) or np.any(counts[0::2] != counts[1::2]):
        raise RuntimeError(f"Incomplete training climatology bins: {counts.tolist()}")
    background = (sums / counts[:, None, None, None]).astype(np.float32)
    BACKGROUND.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        BACKGROUND, background=background, counts=counts, variables=np.asarray(VARIABLES),
        lat=rlat, lon=rlon, buffer_json=json.dumps(BUFFER, sort_keys=True),
        provenance="ERA5 input fields, 1979-2018, monthly x 00/12 UTC means",
    )
    print(f"Wrote {BACKGROUND}")


@lru_cache(maxsize=1)
def load_background() -> np.ndarray:
    data = np.load(BACKGROUND, allow_pickle=False)
    if data["variables"].tolist() != VARIABLES:
        raise RuntimeError("Background variable ordering mismatch")
    return np.asarray(data["background"], dtype=np.float64)


@lru_cache(maxsize=2)
def igra(year: int):
    path = IGRA_2019 if year == 2019 else IGRA_2020
    with path.open("rb") as handle:
        return pickle.load(handle)


def nearest_global_cells(locs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat, lon, _, _, _, _ = grids()
    locs = np.asarray(locs, dtype=np.float64).reshape(-1, 2)
    obs_lon = ((locs[:, 1] + 180.0) % 360.0) - 180.0
    lat_i = np.rint((locs[:, 0] - lat[0]) / (lat[1] - lat[0])).astype(int)
    # lon is discontinuous in storage at 180, so use the original 0--360 index.
    lon360 = obs_lon % 360.0
    lon_i = np.rint(lon360 / 1.40625).astype(int) % len(lon)
    lat_i = np.clip(lat_i, 0, len(lat) - 1)
    return lat_i, lon_i


def source_superobs(locs: np.ndarray, vals: np.ndarray) -> dict[int, float]:
    """Equal-report mean within each nearest ERA5 cell for one source."""
    locs = np.asarray(locs, dtype=np.float64).reshape(-1, 2)
    vals = np.asarray(vals, dtype=np.float64).reshape(-1)
    valid = np.isfinite(vals) & np.all(np.isfinite(locs), axis=1)
    if not np.any(valid):
        return {}
    locs, vals = locs[valid], vals[valid]
    lat_i, lon_i = nearest_global_cells(locs)
    lat, lon, _, _, _, _ = grids()
    lon_obs = lon[lon_i]
    keep = ((lat[lat_i] >= BUFFER["lat_min"]) & (lat[lat_i] <= BUFFER["lat_max"]) &
            (lon_obs >= BUFFER["lon_min"]) & (lon_obs <= BUFFER["lon_max"]))
    flat = lat_i[keep] * len(lon) + lon_i[keep]
    vals = vals[keep]
    sums = defaultdict(float)
    nums = defaultdict(int)
    for cell, value in zip(flat.tolist(), vals.tolist()):
        sums[int(cell)] += float(value)
        nums[int(cell)] += 1
    return {cell: sums[cell] / nums[cell] for cell in sums}


def merge_sources(source_maps: list[dict[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Average source-level superobservations equally where sources share a cell."""
    by_cell: dict[int, list[float]] = defaultdict(list)
    for mapping in source_maps:
        for cell, value in mapping.items():
            by_cell[cell].append(value)
    if not by_cell:
        return np.empty((0, 2)), np.empty(0)
    lat, lon, _, _, _, _ = grids()
    cells = np.asarray(sorted(by_cell), dtype=int)
    lat_i, lon_i = cells // len(lon), cells % len(lon)
    locs = np.column_stack([lat[lat_i], lon[lon_i]])
    vals = np.asarray([np.mean(by_cell[int(cell)]) for cell in cells], dtype=np.float64)
    return locs, vals


def load_observations(year: int, timestep: int, variable_index: int, mode: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    variable = VARIABLES[variable_index]
    mean, std = normalization()
    item = igra(year)[timestep]
    r_locs = np.asarray(item[0][0][variable_index], dtype=np.float64).reshape(-1, 2)
    r_norm = np.asarray(item[1][0][variable_index], dtype=np.float64).reshape(-1)
    r_map = source_superobs(r_locs, r_norm * std[variable_index] + mean[variable_index])
    sources = [r_map]
    counts = {"R_cells": len(r_map), "A_cells": 0, "S_cells": 0}
    if mode == "RAS" and variable in AIRCRAFT_VARS:
        root = AIRCRAFT_2019 if year == 2019 else AIRCRAFT_2020
        path = root / "obs" / f"madis_aircraft_13var_t{timestep:04d}.npz"
        data = np.load(path, allow_pickle=False)
        locs = np.asarray(data[f"{variable}_locs"])
        vals = np.asarray(data[f"{variable}_vals"])
        if f"{variable}_data_source" in data.files:
            keep = np.isin(np.asarray(data[f"{variable}_data_source"], dtype=np.int16), ALLOWED_AIRCRAFT_SOURCES)
            locs, vals = locs[keep], vals[keep]
        a_map = source_superobs(locs, vals)
        sources.append(a_map)
        counts["A_cells"] = len(a_map)
    if mode == "RAS" and variable in SURFACE_VARS:
        root = SURFACE_2019 if year == 2019 else SURFACE_2020
        path = root / "obs" / f"madis_metar_surface_13var_t{timestep:04d}.npz"
        data = np.load(path, allow_pickle=False)
        s_map = source_superobs(data[f"{variable}_locs"], data[f"{variable}_vals"])
        sources.append(s_map)
        counts["S_cells"] = len(s_map)
    locs, vals = merge_sources(sources)
    counts["union_cells"] = len(vals)
    return locs, vals, counts


def regional_background(year: int, timestep: int) -> np.ndarray:
    dt = timestamp(year, timestep)
    return load_background()[background_bin(dt.month, dt.hour)]


def target_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, _, rlat, rlon, _, _ = grids()
    lat_mask = (rlat >= CONUS["lat_min"]) & (rlat <= CONUS["lat_max"])
    lon_mask = (rlon >= CONUS["lon_min"]) & (rlon <= CONUS["lon_max"])
    ii, jj = np.where(lat_mask[:, None] & lon_mask[None, :])
    locs = np.column_stack([rlat[ii], rlon[jj]])
    weights = np.cos(np.deg2rad(rlat[ii]))
    return locs, weights, ii, jj, np.ix_(lat_mask, lon_mask)


def project(locs: np.ndarray) -> np.ndarray:
    """Legacy v1/v2 fixed-standard-parallel projection in kilometres."""
    locs = np.asarray(locs, dtype=np.float64)
    x = EARTH_KM_PER_DEG * math.cos(math.radians(PROJECTION_LAT)) * locs[:, 1]
    y = EARTH_KM_PER_DEG * locs[:, 0]
    return np.column_stack([x, y])


def unit_sphere_xyz(locs: np.ndarray) -> np.ndarray:
    """Convert latitude/longitude degrees to three-dimensional unit vectors."""
    locs = np.asarray(locs, dtype=np.float64)
    lat = np.deg2rad(locs[:, 0])
    lon = np.deg2rad(locs[:, 1])
    cos_lat = np.cos(lat)
    return np.column_stack([
        cos_lat * np.cos(lon),
        cos_lat * np.sin(lon),
        np.sin(lat),
    ])


def great_circle_km(left_xyz: np.ndarray, right_xyz: np.ndarray) -> np.ndarray:
    """Great-circle distance for broadcast-compatible unit-vector arrays."""
    dot = np.sum(left_xyz * right_xyz, axis=-1)
    return EARTH_RADIUS_KM * np.arccos(np.clip(dot, -1.0, 1.0))


def residual_kriging(
    obs_locs: np.ndarray,
    obs_vals: np.ndarray,
    background: np.ndarray,
    variable_index: int,
    length_km: float,
    neighbors: int,
    nugget: float,
) -> np.ndarray:
    target_locs, _, ti, tj, _ = target_geometry()
    base_target = background[variable_index, ti, tj].astype(np.float64)
    if len(obs_vals) == 0:
        return base_target
    _, _, rlat, rlon, lat_slice, lon_slice = grids()
    lat_i, lon_i = nearest_global_cells(obs_locs)
    li = lat_i - int(lat_slice.start)
    lj = lon_i - int(lon_slice.start)
    obs_base = background[variable_index, li, lj]
    residual = np.asarray(obs_vals, dtype=np.float64) - obs_base
    k = min(int(neighbors), len(obs_vals))
    if DISTANCE_METRIC == "great_circle":
        obs_coordinates = unit_sphere_xyz(obs_locs)
        target_coordinates = unit_sphere_xyz(target_locs)
        # Euclidean chord distance on the unit sphere has the same neighbour
        # ordering as angular distance.  Exact great-circle distances are then
        # used in the covariance model below.
        _, indices = cKDTree(obs_coordinates).query(target_coordinates, k=k)
        if k == 1:
            indices = indices[:, None]
        local_coordinates = obs_coordinates[indices]
        distances = great_circle_km(
            local_coordinates, target_coordinates[:, None, :]
        )
        pair = great_circle_km(
            local_coordinates[:, :, None, :],
            local_coordinates[:, None, :, :],
        )
    else:
        obs_coordinates = project(obs_locs)
        target_coordinates = project(target_locs)
        distances, indices = cKDTree(obs_coordinates).query(target_coordinates, k=k)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        local_coordinates = obs_coordinates[indices]
        pair = np.linalg.norm(
            local_coordinates[:, :, None, :] - local_coordinates[:, None, :, :],
            axis=-1,
        )
    cov = np.exp(-pair / float(length_km))
    diag = np.arange(k)
    cov[:, diag, diag] += float(nugget)
    target_cov = np.exp(-distances / float(length_km))
    try:
        weights = np.linalg.solve(cov, target_cov[..., None]).squeeze(-1)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(cov, target_cov[..., None], rcond=1e-8)[0].squeeze(-1)
    increment = np.sum(weights * residual[indices], axis=1)
    # With no observation within three correlation lengths, retain background.
    increment = np.where(distances[:, 0] <= 3.0 * float(length_km), increment, 0.0)
    return base_target + increment


def truth_target(year: int, timestep: int) -> np.ndarray:
    subset = "val" if year == 2019 else "test"
    _, _, _, _, lat_slice, lon_slice = grids()
    _, _, ti, tj, _ = target_geometry()
    path = ERA5 / subset / f"{year}_{timestep:04d}.h5"
    with h5py.File(path, "r") as handle:
        region = np.stack([np.asarray(handle["input"][v][lat_slice, lon_slice], dtype=np.float64) for v in VARIABLES])
    return region[:, ti, tj]


def weighted_rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    _, weights, _, _, _ = target_geometry()
    return float(np.sqrt(np.sum(weights * (prediction - truth) ** 2) / np.sum(weights)))


def candidate_id(length_km: float, neighbors: int, nugget: float) -> str:
    return f"L{int(length_km):04d}_K{int(neighbors):02d}_N{nugget:.2f}"


def evaluate_2019_timestep(timestep: int) -> list[dict[str, object]]:
    background = regional_background(2019, timestep)
    truth = truth_target(2019, timestep)
    rows: list[dict[str, object]] = []
    for vi, variable in enumerate(VARIABLES):
        observations = {}
        for mode in ("R", "RAS"):
            locs, vals, counts = load_observations(2019, timestep, vi, mode)
            observations[mode] = (locs, vals, counts)
        for length in LENGTHS_KM:
            for neighbors in NEIGHBORS:
                for nugget in NUGGETS:
                    cid = candidate_id(length, neighbors, nugget)
                    for mode in ("R", "RAS"):
                        locs, vals, counts = observations[mode]
                        pred = residual_kriging(locs, vals, background, vi, length, neighbors, nugget)
                        rows.append({
                            "year": 2019, "timestep": timestep,
                            "datetime_utc": timestamp(2019, timestep).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "variable": variable, "mode": mode, "candidate_id": cid,
                            "length_km": length, "neighbors": neighbors, "nugget": nugget,
                            "area_weighted_rmse": weighted_rmse(pred, truth[vi]), **counts,
                        })
    return rows


def select_2019(workers: int) -> None:
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, result in enumerate(executor.map(evaluate_2019_timestep, TIMESTEPS_2019), start=1):
            rows.extend(result)
            print(f"[selection] {i}/{len(TIMESTEPS_2019)}", flush=True)
    frame = pd.DataFrame(rows)
    out = RUN / "tables/2019_candidate_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    ras = frame[frame["mode"] == "RAS"]
    ranking = (ras.groupby(["variable", "candidate_id", "length_km", "neighbors", "nugget"], as_index=False)
               .agg(mean_rmse=("area_weighted_rmse", "mean"), median_rmse=("area_weighted_rmse", "median")))
    ranking["rank"] = ranking.groupby("variable")["mean_rmse"].rank(method="min")
    ranking = ranking.sort_values(["variable", "rank", "candidate_id"])
    ranking.to_csv(RUN / "tables/2019_candidate_ranking_by_variable.csv", index=False)
    chosen = ranking[ranking["rank"] == 1].drop_duplicates("variable")
    selected = {
        "selection_year": 2019,
        "selection_timesteps": TIMESTEPS_2019,
        "criterion": "lowest mean cosine-latitude area-weighted strict-CONUS RMSE for kriging R+A+S",
        "kriging_type": "local simple kriging of observation-minus-background residuals",
        "distance_metric": DISTANCE_METRIC,
        "earth_radius_km": EARTH_RADIUS_KM if DISTANCE_METRIC == "great_circle" else None,
        "candidate_grid": {"length_km": LENGTHS_KM, "neighbors": NEIGHBORS, "nugget": NUGGETS},
        "per_variable": {
            row.variable: {"candidate_id": row.candidate_id, "length_km": float(row.length_km),
                           "neighbors": int(row.neighbors), "nugget": float(row.nugget),
                           "mean_2019_ras_rmse": float(row.mean_rmse)}
            for row in chosen.itertuples(index=False)
        },
    }
    SELECTED.write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


def timesteps_2020() -> list[int]:
    payload = json.loads(TIMESTEPS_2020.read_text())
    values = payload.get("timesteps", payload.get("matched_timesteps", []))
    if len(values) != 723 or len(set(values)) != 723:
        raise ValueError(f"Expected 723 unique timesteps, found {len(values)}")
    return [int(x) for x in values]


def evaluate_2020_chunk(chunk_id: int, chunk_size: int) -> None:
    selected = json.loads(SELECTED.read_text())["per_variable"]
    steps = timesteps_2020()
    chunk = steps[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
    if not chunk:
        print(f"Empty chunk {chunk_id}")
        return
    out_dir = RUN / "outputs/2020"
    field_dir = out_dir / "fields"
    field_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, timestep in enumerate(chunk, start=1):
        background = regional_background(2020, timestep)
        truth = truth_target(2020, timestep)
        fields = {"truth": truth.astype(np.float32)}
        for mode in ("R", "RAS"):
            mode_fields = []
            for vi, variable in enumerate(VARIABLES):
                hp = selected[variable]
                locs, vals, counts = load_observations(2020, timestep, vi, mode)
                pred = residual_kriging(locs, vals, background, vi, hp["length_km"], hp["neighbors"], hp["nugget"])
                mode_fields.append(pred.astype(np.float32))
                rows.append({
                    "year": 2020, "timestep": timestep,
                    "datetime_utc": timestamp(2020, timestep).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "variable": variable, "mode": mode, "candidate_id": hp["candidate_id"],
                    "length_km": hp["length_km"], "neighbors": hp["neighbors"], "nugget": hp["nugget"],
                    "area_weighted_rmse": weighted_rmse(pred, truth[vi]), **counts,
                })
            fields[f"kriging_{mode}"] = np.stack(mode_fields)
        np.savez_compressed(field_dir / f"t{timestep:04d}.npz", **fields)
        print(f"[2020 chunk {chunk_id}] {index}/{len(chunk)} t={timestep:04d}", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / f"chunk_{chunk_id:03d}_metrics.csv", index=False)


def aggregate_2020() -> None:
    files = sorted((RUN / "outputs/2020").glob("chunk_*_metrics.csv"))
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    expected = len(timesteps_2020()) * len(VARIABLES) * 2
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} kriging metric rows, found {len(frame)}")
    if frame.duplicated(["timestep", "variable", "mode"]).any():
        raise RuntimeError("Duplicate kriging metric rows")
    frame.to_csv(RUN / "tables/2020_kriging_metrics_by_timestep_variable.csv", index=False)
    pivot = frame.pivot(index=["timestep", "datetime_utc", "variable"], columns="mode", values="area_weighted_rmse").reset_index()
    pivot["kriging_ras_pct_vs_kriging_r"] = 100.0 * (pivot["RAS"] - pivot["R"]) / pivot["R"]

    diffusion = pd.read_csv(DIFFUSION_METRICS)
    diffusion = diffusion[(diffusion["protocol"] == "R+A+S") & (diffusion["region"] == "strict_conus")]
    diffusion = diffusion[["timestep", "variable", "area_weighted_baseline_rmse", "area_weighted_method_rmse"]]
    diffusion = diffusion.rename(columns={
        "area_weighted_baseline_rmse": "diffusion_R_rmse",
        "area_weighted_method_rmse": "diffusion_RAS_rmse",
    })
    paired = pivot.merge(diffusion, on=["timestep", "variable"], validate="one_to_one")
    paired = paired.rename(columns={"R": "kriging_R_rmse", "RAS": "kriging_RAS_rmse"})
    paired["kriging_ras_pct_vs_diffusion_ras"] = 100.0 * (paired["kriging_RAS_rmse"] - paired["diffusion_RAS_rmse"]) / paired["diffusion_RAS_rmse"]
    paired["diffusion_ras_pct_vs_kriging_ras"] = 100.0 * (paired["diffusion_RAS_rmse"] - paired["kriging_RAS_rmse"]) / paired["kriging_RAS_rmse"]
    paired.to_csv(RUN / "tables/2020_kriging_diffusion_paired_metrics.csv", index=False)

    per_var = paired.groupby("variable", as_index=False).agg(
        kriging_R_rmse=("kriging_R_rmse", "mean"), kriging_RAS_rmse=("kriging_RAS_rmse", "mean"),
        diffusion_R_rmse=("diffusion_R_rmse", "mean"), diffusion_RAS_rmse=("diffusion_RAS_rmse", "mean"),
        kriging_R_to_RAS_pct=("kriging_ras_pct_vs_kriging_r", "mean"),
        diffusion_RAS_vs_kriging_RAS_pct=("diffusion_ras_pct_vs_kriging_ras", "mean"),
    )
    per_var["unit"] = per_var["variable"].map(UNITS)
    per_var.to_csv(RUN / "tables/2020_method_comparison_by_variable.csv", index=False)
    by_time = paired.groupby(["timestep", "datetime_utc"], as_index=False).agg(
        kriging_R_to_RAS_pct=("kriging_ras_pct_vs_kriging_r", "mean"),
        diffusion_RAS_vs_kriging_RAS_pct=("diffusion_ras_pct_vs_kriging_ras", "mean"),
    )
    by_time.to_csv(RUN / "tables/2020_method_comparison_by_time.csv", index=False)

    def moving_block_interval(values: np.ndarray, block: int = 28, replicates: int = 10_000) -> dict[str, float]:
        values = np.asarray(values, dtype=float)
        rng = np.random.default_rng(1701)
        n = len(values)
        blocks_needed = int(np.ceil(n / block))
        draws = np.empty(replicates, dtype=float)
        offsets = np.arange(block)
        for rep in range(replicates):
            starts = rng.integers(0, n, size=blocks_needed)
            indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
            draws[rep] = np.mean(values[indices])
        low, high = np.quantile(draws, [0.025, 0.975])
        return {"mean": float(np.mean(values)), "ci95_low": float(low), "ci95_high": float(high)}

    intervals = {
        "kriging_R_to_RAS_pct": moving_block_interval(by_time["kriging_R_to_RAS_pct"].to_numpy()),
        "diffusion_RAS_vs_kriging_RAS_pct": moving_block_interval(
            by_time["diffusion_RAS_vs_kriging_RAS_pct"].to_numpy()
        ),
        "block_length_matched_12h_cases": 28,
        "bootstrap_replicates": 10_000,
        "seed": 1701,
    }
    (RUN / "outputs/2020_moving_block_intervals.json").write_text(json.dumps(intervals, indent=2) + "\n")
    summary = {
        "n_timesteps": int(paired["timestep"].nunique()),
        "n_variables": int(paired["variable"].nunique()),
        "mean_kriging_R_to_RAS_pct_equal_variable_time": float(paired["kriging_ras_pct_vs_kriging_r"].mean()),
        "mean_diffusion_RAS_vs_kriging_RAS_pct_equal_variable_time": float(paired["diffusion_ras_pct_vs_kriging_ras"].mean()),
        "median_diffusion_RAS_vs_kriging_RAS_pct": float(paired["diffusion_ras_pct_vs_kriging_ras"].median()),
        "fraction_variable_times_diffusion_better_than_kriging": float((paired["diffusion_RAS_rmse"] < paired["kriging_RAS_rmse"]).mean()),
    }
    (RUN / "outputs/2020_comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_summary_figure(per_var)
    write_report(summary, intervals, per_var)
    print(json.dumps(summary, indent=2))


def make_summary_figure(per_var: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "T2m", "U10", "V10", "Z500", "Z850", "U500", "U850",
        "V500", "V850", "T500", "T850", "Q500", "Q850",
    ]
    frame = per_var.set_index("variable").loc[VARIABLES].reset_index()
    y = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.8), sharey=True)
    panels = [
        ("kriging_R_to_RAS_pct", "Kriging: R+A+S relative to R"),
        ("diffusion_RAS_vs_kriging_RAS_pct", "Diffusion relative to kriging (same R+A+S)"),
    ]
    for ax, (column, title) in zip(axes, panels):
        values = frame[column].to_numpy(float)
        colors = np.where(values <= 0.0, "#2166ac", "#b2182b")
        ax.barh(y, values, color=colors, alpha=0.9)
        if column == "kriging_R_to_RAS_pct":
            zero = np.isclose(values, 0.0, atol=1e-12)
            ax.scatter(np.zeros(np.sum(zero)), y[zero], facecolors="white", edgecolors="#2166ac",
                       linewidths=1.2, s=34, zorder=3, label="No A/S observations for variable")
            ax.legend(loc="lower left", frameon=False, fontsize=8)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Mean RMSE change (%)\nnegative is lower RMSE")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("2020 strict-CONUS area-weighted RMSE comparison", fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(RUN / "figures" / f"kriging_diffusion_comparison.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: dict[str, float], intervals: dict[str, object], per_var: pd.DataFrame) -> None:
    d = intervals["diffusion_RAS_vs_kriging_RAS_pct"]
    k = intervals["kriging_R_to_RAS_pct"]
    best = per_var.sort_values("diffusion_RAS_vs_kriging_RAS_pct").iloc[0]
    worst = per_var.sort_values("diffusion_RAS_vs_kriging_RAS_pct").iloc[-1]
    text = f"""# Residual-kriging baseline: completed result

## Scope

This experiment addresses Reviewer 2 Major Comment 1.  It compares the
diffusion reconstruction with a non-diffusion fixed-background residual
kriging baseline using the same observation-source endpoint.  It does not
answer Minor Comment 4 and does not treat A-only/S-only/prior-only experiments
as substitutes for the method-class comparison.

## Frozen protocol

- Background: 1979--2018 ERA5 training climatology, monthly by 00/12 UTC.
- Development: 24 prespecified 2019 times; hyperparameters selected per
  variable using R+A+S area-weighted CONUS RMSE.
- Evaluation: {summary['n_timesteps']} matched 2020 times and
  {summary['n_variables']} variables.
- Observation handling: source-specific equal-cell superobservations, followed
  by equal source averaging when multiple sources occupy the same cell.
- Kriging: local exponential-covariance simple kriging of residuals, using
  `{DISTANCE_METRIC}` distances.

## Main paired results

- Within the classical baseline, adding A and S to R changed RMSE by
  **{k['mean']:.3f}%** (28-matched-case moving-block 95% interval
  **[{k['ci95_low']:.3f}, {k['ci95_high']:.3f}]%**).
- With the same R+A+S observation endpoint, diffusion changed RMSE relative to
  kriging by **{d['mean']:.3f}%** (28-matched-case moving-block 95% interval
  **[{d['ci95_low']:.3f}, {d['ci95_high']:.3f}]%**).  Negative values mean the
  diffusion reconstruction has lower RMSE.
- Diffusion had lower RMSE in
  **{100.0 * summary['fraction_variable_times_diffusion_better_than_kriging']:.2f}%**
  of paired variable-times.
- The largest mean diffusion advantage was for `{best['variable']}`
  ({best['diffusion_RAS_vs_kriging_RAS_pct']:.3f}%); the weakest was for
  `{worst['variable']}` ({worst['diffusion_RAS_vs_kriging_RAS_pct']:.3f}%).

These statements are descriptive evidence against the explicitly frozen
baseline.  They should not be generalized to all kriging/optimal-interpolation
implementations or operational data-assimilation systems.

## Primary artifacts

- `tables/2020_kriging_diffusion_paired_metrics.csv`
- `tables/2020_method_comparison_by_variable.csv`
- `tables/2020_method_comparison_by_time.csv`
- `outputs/2019_selected_hyperparameters.json`
- `outputs/2020_moving_block_intervals.json`
- `figures/kriging_diffusion_comparison.pdf`
"""
    (RUN / "REPORT.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["background", "select", "evaluate", "aggregate"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=30)
    args = parser.parse_args()
    required_environment = (
        "KRIGING_RUN_ROOT", "ERA5_ROOT", "IGRA_2019_PKL",
        "AIRCRAFT_2019_ROOT", "SURFACE_2019_ROOT", "IGRA_2020_PKL",
        "AIRCRAFT_2020_ROOT", "SURFACE_2020_ROOT",
        "EVALUATION_TIMESTEP_MANIFEST", "AREA_WEIGHTED_DIFFUSION_METRICS_CSV",
    )
    missing = [name for name in required_environment if not os.environ.get(name)]
    if missing:
        parser.error("set the following environment variables: " + ", ".join(missing))
    if args.stage == "background":
        build_background()
    elif args.stage == "select":
        select_2019(args.workers)
    elif args.stage == "evaluate":
        evaluate_2020_chunk(args.chunk_id, args.chunk_size)
    else:
        aggregate_2020()


if __name__ == "__main__":
    main()
