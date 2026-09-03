#!/usr/bin/env python3
"""Quick coverage audit for MADIS aircraft netCDF(.gz) samples.

This is a first-pass audit only. MADIS sample files inspected so far expose
pressure altitude / GPS altitude rather than a direct pressure variable, so this
script uses the standard-atmosphere pressure-altitude approximation to estimate
500/850 hPa candidate counts. Final protocol should revisit this after a deeper
metadata/QC check.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


STRICT_CONUS = (24.0, 50.0, -125.0, -66.0)
NORTH_AMERICA = (10.0, 70.0, -170.0, -50.0)


def decode_to_temp(path: Path) -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="madis_aircraft_audit_", suffix=".nc", delete=False)
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
            return arr
    return None


def pressure_hpa_from_altitude_m(alt_m: np.ndarray) -> np.ndarray:
    """Approximate pressure from pressure altitude using ISA troposphere."""
    alt = np.asarray(alt_m, dtype=np.float64)
    pressure = 1013.25 * np.power(np.maximum(0.0, 1.0 - 2.25577e-5 * alt), 5.25588)
    pressure = np.where(np.isfinite(alt), pressure, np.nan)
    return pressure


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
    track_lat = read_var(ds, "trackLat")
    track_lon = read_var(ds, "trackLon")
    if track_lat is not None and track_lon is not None and track_lat.shape == target_shape and track_lon.shape == target_shape:
        return track_lat, track_lon, "trackLat_trackLon"
    lat = read_var(ds, "latitude", "LAT")
    lon = read_var(ds, "longitude", "LON")
    if lat is None or lon is None:
        return None, None, "missing"
    lat2, lon2 = broadcast_profile_locations(lat, lon, target_shape)
    return lat2, lon2, "latitude_longitude_broadcast"


def count_window(mask_base: np.ndarray, pressure: np.ndarray, lo: float, hi: float) -> int:
    return int(np.isfinite(pressure[mask_base]).sum() if False else np.sum(mask_base & (pressure >= lo) & (pressure <= hi)))


def audit_file(path: Path) -> dict[str, Any]:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise SystemExit("netCDF4 is required in the active Python environment") from exc

    tmp = decode_to_temp(path)
    try:
        with Dataset(tmp, "r") as ds:
            temp = read_var(ds, "temperature", "T")
            wind_speed = read_var(ds, "windSpeed", "FF")
            wind_dir = read_var(ds, "windDir", "DD")
            rel_humidity = read_var(ds, "relHumidity", "RH")
            dewpoint = read_var(ds, "dewpoint", "TD")
            altitude = read_var(ds, "altitude", "HT", "GPSaltitude", "GPSHT")
            time_obs = read_var(ds, "timeObs", "TDAYSEC")
            if temp is None or altitude is None:
                raise RuntimeError(f"Missing core vars in {path}")
            lat2, lon2, location_source = read_locations(ds, temp.shape)
            if lat2 is None or lon2 is None:
                raise RuntimeError(f"Missing location vars in {path}")
            pressure = pressure_hpa_from_altitude_m(altitude)
            u, v = to_uv(wind_speed, wind_dir)

            valid_loc = np.isfinite(lat2) & np.isfinite(lon2)
            conus = valid_loc & in_box(lat2, lon2, STRICT_CONUS)
            na = valid_loc & in_box(lat2, lon2, NORTH_AMERICA)

            valid_temp = conus & np.isfinite(temp)
            valid_wind = conus & np.isfinite(u) & np.isfinite(v) if u is not None else np.zeros_like(conus, dtype=bool)
            valid_rh = conus & np.isfinite(rel_humidity) if rel_humidity is not None else np.zeros_like(conus, dtype=bool)
            valid_td = conus & np.isfinite(dewpoint) if dewpoint is not None else np.zeros_like(conus, dtype=bool)

            return {
                "file": str(path),
                "product": "acarsProfiles" if "acarsProfiles" in str(path) else "acars",
                "location_source": location_source,
                "shape": "x".join(map(str, temp.shape)),
                "n_records_or_values": int(temp.size),
                "n_conus_location_values": int(np.sum(conus)),
                "n_north_america_location_values": int(np.sum(na)),
                "n_conus_temp_finite": int(np.sum(valid_temp)),
                "n_conus_wind_finite": int(np.sum(valid_wind)),
                "n_conus_rh_finite": int(np.sum(valid_rh)),
                "n_conus_dewpoint_finite": int(np.sum(valid_td)),
                "n_conus_t500_strict_alt_derived": count_window(valid_temp, pressure, 475.0, 525.0),
                "n_conus_t850_strict_alt_derived": count_window(valid_temp, pressure, 825.0, 875.0),
                "n_conus_uv500_strict_alt_derived": count_window(valid_wind, pressure, 475.0, 525.0),
                "n_conus_uv850_strict_alt_derived": count_window(valid_wind, pressure, 825.0, 875.0),
                "n_conus_t500_wide_alt_derived": count_window(valid_temp, pressure, 450.0, 550.0),
                "n_conus_t850_wide_alt_derived": count_window(valid_temp, pressure, 800.0, 900.0),
                "n_conus_uv500_wide_alt_derived": count_window(valid_wind, pressure, 450.0, 550.0),
                "n_conus_uv850_wide_alt_derived": count_window(valid_wind, pressure, 800.0, 900.0),
                "lat_min": float(np.nanmin(lat2[valid_loc])) if np.any(valid_loc) else np.nan,
                "lat_max": float(np.nanmax(lat2[valid_loc])) if np.any(valid_loc) else np.nan,
                "lon_min": float(np.nanmin(lon2[valid_loc])) if np.any(valid_loc) else np.nan,
                "lon_max": float(np.nanmax(lon2[valid_loc])) if np.any(valid_loc) else np.nan,
                "pressure_min_hpa_alt_derived": float(np.nanmin(pressure)),
                "pressure_max_hpa_alt_derived": float(np.nanmax(pressure)),
                "pressure_median_hpa_alt_derived": float(np.nanmedian(pressure)),
                "time_min": float(np.nanmin(time_obs)) if time_obs is not None else np.nan,
                "time_max": float(np.nanmax(time_obs)) if time_obs is not None else np.nan,
            }
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--csv-out", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    rows = [audit_file(path) for path in args.files]
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    args.json_out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.json_out}")
    for row in rows:
        print(
            Path(row["file"]).name,
            row["product"],
            "CONUS temp",
            row["n_conus_temp_finite"],
            "CONUS wind",
            row["n_conus_wind_finite"],
            "t850 strict",
            row["n_conus_t850_strict_alt_derived"],
            "uv850 strict",
            row["n_conus_uv850_strict_alt_derived"],
        )


if __name__ == "__main__":
    main()
