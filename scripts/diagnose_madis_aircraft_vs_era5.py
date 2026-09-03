#!/usr/bin/env python3
"""Compare MADIS aircraft observations against matching ERA5 pressure levels.

This is a feasibility diagnostic, not the final aircraft observation protocol.
The inspected MADIS sample exposes pressure altitude rather than a direct
pressure variable, so pressure windows are derived from a standard-atmosphere
conversion of pressure altitude.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


STRICT_CONUS = (24.0, 50.0, -125.0, -66.0)
NORTH_AMERICA = (10.0, 70.0, -170.0, -50.0)
GLOBAL = (-90.0, 90.0, -180.0, 180.0)
START_2020 = datetime(2020, 1, 1, 0, 0)


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    era5_name: str
    lo_hpa: float
    hi_hpa: float
    obs_kind: str
    level_hpa: int


SPECS = [
    CandidateSpec("t500_strict", "temperature_500", 475.0, 525.0, "temperature", 500),
    CandidateSpec("t850_strict", "temperature_850", 825.0, 875.0, "temperature", 850),
    CandidateSpec("u500_strict", "u_component_of_wind_500", 475.0, 525.0, "u", 500),
    CandidateSpec("v500_strict", "v_component_of_wind_500", 475.0, 525.0, "v", 500),
    CandidateSpec("u850_strict", "u_component_of_wind_850", 825.0, 875.0, "u", 850),
    CandidateSpec("v850_strict", "v_component_of_wind_850", 825.0, 875.0, "v", 850),
]


def decode_to_temp(path: Path) -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="madis_aircraft_vs_era5_", suffix=".nc", delete=False)
    tmp_path = Path(tmp.name)
    with gzip.open(path, "rb") as src:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
    tmp.close()
    return tmp_path


def read_var(ds: Any, *names: str) -> np.ndarray | None:
    for name in names:
        if name in ds.variables:
            var = ds.variables[name]
            arr = np.asarray(var[:], dtype=np.float64)
            fill = getattr(var, "_FillValue", None)
            if fill is not None:
                arr = np.where(arr == float(fill), np.nan, arr)
            missing = getattr(var, "missing_value", None)
            if missing is not None:
                try:
                    arr = np.where(arr == float(missing), np.nan, arr)
                except Exception:
                    pass
            arr = np.where(arr <= -9990.0, np.nan, arr)
            arr = np.where(arr >= 99990.0, np.nan, arr)
            return arr
    return None


def read_qcr(ds: Any, name: str, target_shape: tuple[int, ...]) -> np.ndarray:
    qname = f"{name}QCR"
    if qname not in ds.variables:
        return np.zeros(target_shape, dtype=np.int64)
    arr = np.asarray(ds.variables[qname][:])
    if arr.shape == target_shape:
        return arr.astype(np.int64)
    if len(target_shape) == 2 and arr.ndim == 1 and arr.shape[0] == target_shape[0]:
        return np.repeat(arr[:, None], target_shape[1], axis=1).astype(np.int64)
    return np.broadcast_to(arr, target_shape).astype(np.int64)


def pressure_hpa_from_altitude_m(alt_m: np.ndarray) -> np.ndarray:
    alt = np.asarray(alt_m, dtype=np.float64)
    pressure = 1013.25 * np.power(np.maximum(0.0, 1.0 - 2.25577e-5 * alt), 5.25588)
    return np.where(np.isfinite(alt), pressure, np.nan)


def to_uv(speed: np.ndarray | None, direction: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if speed is None or direction is None:
        return None, None
    rad = np.deg2rad(direction)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v


def in_box(lat: np.ndarray, lon: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = box
    return (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)


def broadcast_profile_locations(lat: np.ndarray, lon: np.ndarray, target_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    if lat.shape == target_shape:
        return lat, lon
    if len(target_shape) == 2 and lat.ndim == 1 and lat.shape[0] == target_shape[0]:
        return np.repeat(lat[:, None], target_shape[1], axis=1), np.repeat(lon[:, None], target_shape[1], axis=1)
    return np.broadcast_to(lat, target_shape), np.broadcast_to(lon, target_shape)


def read_locations(ds: Any, target_shape: tuple[int, ...]) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Read point locations, preferring profile track coordinates when present."""
    track_lat = read_var(ds, "trackLat")
    track_lon = read_var(ds, "trackLon")
    if track_lat is not None and track_lon is not None and track_lat.shape == target_shape and track_lon.shape == target_shape:
        return track_lat, track_lon, "trackLat_trackLon"
    lat = read_var(ds, "latitude")
    lon = read_var(ds, "longitude")
    if lat is None or lon is None:
        return None, None, "missing"
    lat2, lon2 = broadcast_profile_locations(lat, lon, target_shape)
    return lat2, lon2, "latitude_longitude_broadcast"


def parse_datetime(path: Path) -> datetime:
    m = re.search(r"(\d{8})_(\d{4})\.gz$", path.name)
    if not m:
        raise ValueError(f"Cannot parse datetime from {path}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")


def era5_timestep_index(dt: datetime) -> int:
    hours = (dt - START_2020).total_seconds() / 3600.0
    idx = int(round(hours / 6.0))
    if abs(idx * 6.0 - hours) > 1e-6:
        raise ValueError(f"{dt} is not aligned to a 6-hour ERA5 timestep")
    return idx


def load_era5_field(era5_root: Path, timestep_index: int, variable: str) -> np.ndarray:
    path = era5_root / "test" / f"2020_{timestep_index:04d}.h5"
    with h5py.File(path, "r") as f:
        return np.asarray(f["input"][variable][:], dtype=np.float64)


def interp_regular_grid(field: np.ndarray, lat_grid: np.ndarray, lon_grid: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear interpolation on the ERA5 128x256 regular grid."""
    lats = np.asarray(lat_grid, dtype=np.float64)
    lons = np.asarray(lon_grid, dtype=np.float64)
    vals_lat = field
    # The repo's grid convention is lat/lon arrays plus field[lat, lon].
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        vals_lat = vals_lat[::-1, :]
    lon_in = np.asarray(lon, dtype=np.float64)
    if lons.min() >= 0.0:
        lon_mod = np.mod(lon_in, 360.0)
    else:
        lon_mod = ((lon_in + 180.0) % 360.0) - 180.0
    lat_in = np.asarray(lat, dtype=np.float64)

    i = np.searchsorted(lats, lat_in) - 1
    j = np.searchsorted(lons, lon_mod) - 1
    i = np.clip(i, 0, len(lats) - 2)
    j = np.clip(j, 0, len(lons) - 2)
    lat0, lat1 = lats[i], lats[i + 1]
    lon0, lon1 = lons[j], lons[j + 1]
    wi = np.where(lat1 != lat0, (lat_in - lat0) / (lat1 - lat0), 0.0)
    wj = np.where(lon1 != lon0, (lon_mod - lon0) / (lon1 - lon0), 0.0)
    f00 = vals_lat[i, j]
    f10 = vals_lat[i + 1, j]
    f01 = vals_lat[i, j + 1]
    f11 = vals_lat[i + 1, j + 1]
    return (1 - wi) * (1 - wj) * f00 + wi * (1 - wj) * f10 + (1 - wi) * wj * f01 + wi * wj * f11


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return math.nan
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def audit_file(path: Path, era5_root: Path, lat_grid: np.ndarray, lon_grid: np.ndarray, region: str) -> list[dict[str, Any]]:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise SystemExit("netCDF4 is required in the active Python environment") from exc

    boxes = {
        "strict_conus": STRICT_CONUS,
        "north_america": NORTH_AMERICA,
        "global": GLOBAL,
    }
    box = boxes[region]
    dt = parse_datetime(path)
    t_index = era5_timestep_index(dt)
    product = "acarsProfiles" if "acarsProfiles" in str(path) else "acars"

    tmp = decode_to_temp(path)
    try:
        with Dataset(tmp, "r") as ds:
            temp = read_var(ds, "temperature")
            wind_speed = read_var(ds, "windSpeed")
            wind_dir = read_var(ds, "windDir")
            altitude = read_var(ds, "altitude", "GPSaltitude")
            if temp is None or altitude is None:
                raise RuntimeError(f"Missing core vars in {path}")
            lat2, lon2, location_source = read_locations(ds, temp.shape)
            if lat2 is None or lon2 is None:
                raise RuntimeError(f"Missing location vars in {path}")
            pressure = pressure_hpa_from_altitude_m(altitude)
            u, v = to_uv(wind_speed, wind_dir)

            loc_qc = (
                (read_qcr(ds, "latitude", temp.shape) == 0)
                & (read_qcr(ds, "longitude", temp.shape) == 0)
                & (read_qcr(ds, "altitude", temp.shape) == 0)
            )
            temp_qc = read_qcr(ds, "temperature", temp.shape) == 0
            wind_qc = (read_qcr(ds, "windSpeed", temp.shape) == 0) & (read_qcr(ds, "windDir", temp.shape) == 0)
            base = (
                loc_qc
                & np.isfinite(lat2)
                & np.isfinite(lon2)
                & np.isfinite(pressure)
                & in_box(lat2, lon2, box)
                & (pressure > 100.0)
                & (pressure < 1050.0)
            )

            obs_by_kind = {
                "temperature": temp,
                "u": u if u is not None else np.full_like(temp, np.nan),
                "v": v if v is not None else np.full_like(temp, np.nan),
            }
            qc_by_kind = {
                "temperature": temp_qc & (temp > 180.0) & (temp < 330.0),
                "u": wind_qc & np.isfinite(u) & np.isfinite(v) & (wind_speed >= 0.0) & (wind_speed < 150.0) if u is not None else np.zeros_like(base),
                "v": wind_qc & np.isfinite(u) & np.isfinite(v) & (wind_speed >= 0.0) & (wind_speed < 150.0) if v is not None else np.zeros_like(base),
            }

            rows = []
            era5_cache: dict[str, np.ndarray] = {}
            for spec in SPECS:
                obs = obs_by_kind[spec.obs_kind]
                mask = base & qc_by_kind[spec.obs_kind] & np.isfinite(obs) & (pressure >= spec.lo_hpa) & (pressure <= spec.hi_hpa)
                n = int(np.sum(mask))
                if n == 0:
                    rows.append(empty_row(path, product, dt, t_index, region, spec))
                    continue
                if spec.era5_name not in era5_cache:
                    era5_cache[spec.era5_name] = load_era5_field(era5_root, t_index, spec.era5_name)
                obs_vals = obs[mask].astype(np.float64)
                era5_vals = interp_regular_grid(era5_cache[spec.era5_name], lat_grid, lon_grid, lat2[mask], lon2[mask])
                diff = obs_vals - era5_vals
                rows.append(
                    {
                        "file": str(path),
                        "product": product,
                        "datetime": dt.isoformat(),
                        "era5_timestep_index": t_index,
                        "region": region,
                        "variable": spec.era5_name,
                        "label": spec.label,
                        "level_hpa": spec.level_hpa,
                        "pressure_window_lo_hpa": spec.lo_hpa,
                        "pressure_window_hi_hpa": spec.hi_hpa,
                        "n_obs": n,
                        "obs_mean": float(np.nanmean(obs_vals)),
                        "era5_mean": float(np.nanmean(era5_vals)),
                        "bias_obs_minus_era5": float(np.nanmean(diff)),
                        "mae": float(np.nanmean(np.abs(diff))),
                        "rmse": float(np.sqrt(np.nanmean(diff**2))),
                        "corr": corrcoef_safe(obs_vals, era5_vals),
                        "obs_std": float(np.nanstd(obs_vals)),
                        "era5_std": float(np.nanstd(era5_vals)),
                        "pressure_mean_hpa_alt_derived": float(np.nanmean(pressure[mask])),
                        "pressure_std_hpa_alt_derived": float(np.nanstd(pressure[mask])),
                        "location_source": location_source,
                        "lat_min": float(np.nanmin(lat2[mask])),
                        "lat_max": float(np.nanmax(lat2[mask])),
                        "lon_min": float(np.nanmin(lon2[mask])),
                        "lon_max": float(np.nanmax(lon2[mask])),
                    }
                )
            return rows
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def empty_row(path: Path, product: str, dt: datetime, t_index: int, region: str, spec: CandidateSpec) -> dict[str, Any]:
    return {
        "file": str(path),
        "product": product,
        "datetime": dt.isoformat(),
        "era5_timestep_index": t_index,
        "region": region,
        "variable": spec.era5_name,
        "label": spec.label,
        "level_hpa": spec.level_hpa,
        "pressure_window_lo_hpa": spec.lo_hpa,
        "pressure_window_hi_hpa": spec.hi_hpa,
        "n_obs": 0,
        "obs_mean": math.nan,
        "era5_mean": math.nan,
        "bias_obs_minus_era5": math.nan,
        "mae": math.nan,
        "rmse": math.nan,
        "corr": math.nan,
        "obs_std": math.nan,
        "era5_std": math.nan,
        "pressure_mean_hpa_alt_derived": math.nan,
        "pressure_std_hpa_alt_derived": math.nan,
        "location_source": "none",
        "lat_min": math.nan,
        "lat_max": math.nan,
        "lon_min": math.nan,
        "lon_max": math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--era5-root", type=Path, default=Path("/depot/rmaulik/data/yangxu/era5_subset"))
    parser.add_argument("--region", choices=["strict_conus", "north_america", "global"], default="strict_conus")
    parser.add_argument("--csv-out", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    lat_grid = np.load(args.era5_root / "lat.npy").astype(np.float64)
    lon_grid = np.load(args.era5_root / "lon.npy").astype(np.float64)

    rows: list[dict[str, Any]] = []
    for path in args.files:
        rows.extend(audit_file(path, args.era5_root, lat_grid, lon_grid, args.region))

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    args.json_out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
