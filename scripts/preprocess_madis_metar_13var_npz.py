#!/usr/bin/env python3
"""Build model-ready METAR t2m/u10/v10 NPZ files for an arbitrary year."""

from __future__ import annotations

import argparse
import gzip
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from independent_year_common import STRICT_CONUS, lon_to_180, six_hour_index


def decode_nc(path: Path) -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="madis_metar_", suffix=".nc", delete=False)
    tmp_path = Path(tmp.name)
    with gzip.open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            tmp.write(chunk)
    tmp.close()
    return tmp_path


def data_var(ds: Dataset, name: str, *, filter_extreme_missing: bool = True) -> np.ndarray:
    var = ds.variables[name]
    raw = var[:]
    values = np.asarray(raw.filled(np.nan) if np.ma.isMaskedArray(raw) else raw, dtype=np.float64)
    for attribute in ("_FillValue", "missing_value"):
        missing = getattr(var, attribute, None)
        if missing is not None:
            try:
                values = np.where(values == float(missing), np.nan, values)
            except (TypeError, ValueError):
                pass
    if filter_extreme_missing:
        values = np.where((values <= -9990.0) | (values >= 99990.0), np.nan, values)
    return values


def qcr_var(ds: Dataset, name: str, n: int) -> np.ndarray:
    qname = f"{name}QCR"
    if qname not in ds.variables:
        return np.full(n, 999, dtype=np.int64)
    raw = ds.variables[qname][:]
    return np.asarray(raw.filled(999) if np.ma.isMaskedArray(raw) else raw, dtype=np.int64)


def char_array_to_str(values: np.ndarray) -> np.ndarray:
    output = []
    for row in values:
        if np.ma.isMaskedArray(row):
            row = row.filled(b"")
        output.append(b"".join(row.tolist()).decode("ascii", errors="ignore").strip())
    return np.asarray(output, dtype=object)


def to_uv(speed: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(direction)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def parse_datetime(path: Path) -> datetime:
    token = path.stem.split(".")[0]
    return datetime.strptime(token, "%Y%m%d_%H%M")


def strict_conus_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat0, lat1, lon0, lon1 = STRICT_CONUS
    lon = lon_to_180(lon)
    return (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1)


def build_one(path: Path, output_root: Path, time_window_min: float) -> dict:
    dt = parse_datetime(path)
    timestep = six_hour_index(dt)
    tmp = decode_nc(path)
    try:
        with Dataset(tmp) as ds:
            n = len(ds.dimensions["recNum"])
            station = char_array_to_str(ds.variables["stationName"][:])
            lat = data_var(ds, "latitude")
            lon = lon_to_180(data_var(ds, "longitude"))
            time_obs = data_var(ds, "timeObs", filter_extreme_missing=False)
            offset_min = np.asarray(
                [
                    (datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None) - dt).total_seconds() / 60.0
                    if np.isfinite(value)
                    else np.nan
                    for value in time_obs
                ]
            )
            temperature = data_var(ds, "temperature")
            direction = data_var(ds, "windDir")
            speed = data_var(ds, "windSpeed")
            temperature_qcr = qcr_var(ds, "temperature", n)
            direction_qcr = qcr_var(ds, "windDir", n)
            speed_qcr = qcr_var(ds, "windSpeed", n)
    finally:
        tmp.unlink(missing_ok=True)

    base = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(offset_min)
        & (np.abs(offset_min) <= time_window_min)
        & strict_conus_mask(lat, lon)
    )
    temperature_mask = base & (temperature_qcr == 0) & np.isfinite(temperature) & (temperature >= 180.0) & (temperature <= 330.0)
    wind_mask = (
        base
        & (direction_qcr == 0)
        & (speed_qcr == 0)
        & np.isfinite(direction)
        & np.isfinite(speed)
        & (speed >= 0.0)
        & (speed <= 75.0)
    )
    u_wind, v_wind = to_uv(speed, direction)
    output_dir = output_root / "obs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"madis_metar_surface_13var_t{timestep:04d}.npz"
    metadata = {
        "raw_file": str(path.resolve()),
        "timestep": timestep,
        "datetime_utc": dt.isoformat() + "Z",
        "support": "strict_conus",
        "time_window_min": time_window_min,
        "qcr_rule": "temperatureQCR==0; windDirQCR==0 and windSpeedQCR==0",
        "n_t2m": int(temperature_mask.sum()),
        "n_wind": int(wind_mask.sum()),
        "n_t2m_stations": len(set(station[temperature_mask].tolist())),
        "n_wind_stations": len(set(station[wind_mask].tolist())),
    }
    np.savez_compressed(
        output_path,
        **{
            "2m_temperature_locs": np.stack([lat[temperature_mask], lon[temperature_mask]], axis=1).astype(np.float32),
            "2m_temperature_vals": temperature[temperature_mask].astype(np.float32),
            "10m_u_component_of_wind_locs": np.stack([lat[wind_mask], lon[wind_mask]], axis=1).astype(np.float32),
            "10m_u_component_of_wind_vals": u_wind[wind_mask].astype(np.float32),
            "10m_v_component_of_wind_locs": np.stack([lat[wind_mask], lon[wind_mask]], axis=1).astype(np.float32),
            "10m_v_component_of_wind_vals": v_wind[wind_mask].astype(np.float32),
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        },
    )
    return {**metadata, "processed_file": str(output_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--time-window-min", type=float, default=60.0)
    args = parser.parse_args()

    years = {parse_datetime(path).year for path in args.files}
    if len(years) != 1:
        raise ValueError(f"All input files must belong to one year, got {sorted(years)}")
    rows = [build_one(path, args.out_root, args.time_window_min) for path in sorted(args.files)]
    manifest = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "year": next(iter(years)),
        "output_root": str(args.out_root.resolve()),
        "support": "strict_conus",
        "time_window_min": args.time_window_min,
        "timesteps": [row["timestep"] for row in rows],
        "rows": rows,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} processed METAR files under {args.out_root}")


if __name__ == "__main__":
    main()
