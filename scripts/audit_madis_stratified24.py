#!/usr/bin/env python3
"""Full source, QC, time, support, and pressure-window audit for MADIS."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import tempfile
from collections import Counter
from datetime import timezone
from pathlib import Path

import netCDF4
import numpy as np

from independent_year_common import ROOT, STRICT_CONUS, lon_to_180, stratified24


SOURCE_ATTR_PREFIX = "value_"
ABO_QCR_FIELDS = (
    "timeObsQCR", "latitudeQCR", "longitudeQCR", "altitudeQCR",
    "temperatureQCR", "windSpeedQCR", "windDirQCR",
)
METAR_QCR_FIELDS = ("temperatureQCR", "windSpeedQCR", "windDirQCR")
PRESSURE_WINDOWS = {
    "around5_500": (495.0, 505.0),
    "around5_850": (845.0, 855.0),
    "around25_500": (475.0, 525.0),
    "around25_850": (825.0, 875.0),
}


def array(ds: netCDF4.Dataset, name: str, *, fill=np.nan) -> np.ndarray:
    raw = ds.variables[name][:]
    if np.ma.isMaskedArray(raw):
        raw = raw.filled(fill)
    return np.asarray(raw)


def qcr(ds: netCDF4.Dataset, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in ds.variables:
        return np.full(shape, -999, dtype=np.int32)
    return array(ds, name, fill=-999).astype(np.int32)


def pressure_hpa_from_altitude_m(altitude: np.ndarray) -> np.ndarray:
    alt = np.asarray(altitude, dtype=np.float64)
    value = 1013.25 * np.power(np.maximum(0.0, 1.0 - 2.25577e-5 * alt), 5.25588)
    return np.where(np.isfinite(alt), value, np.nan)


def in_strict_conus(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat0, lat1, lon0, lon1 = STRICT_CONUS
    return (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1)


def decode_station_names(raw: np.ndarray) -> np.ndarray:
    if raw.dtype.kind == "S" and raw.ndim == 2:
        return np.asarray([b"".join(row).decode("ascii", "ignore").strip() for row in raw])
    return raw.astype(str)


def source_mapping(ds: netCDF4.Dataset) -> dict[int, str]:
    variable = ds.variables["dataSource"]
    mapping = {}
    for name in variable.ncattrs():
        if name.startswith(SOURCE_ATTR_PREFIX):
            mapping[int(name[len(SOURCE_ATTR_PREFIX):])] = str(getattr(variable, name))
    return dict(sorted(mapping.items()))


def inspect_abo(path: Path, nominal_epoch: float):
    compressed = path.read_bytes()
    with tempfile.NamedTemporaryFile(suffix=".nc") as handle:
        handle.write(gzip.decompress(compressed))
        handle.flush()
        with netCDF4.Dataset(handle.name) as ds:
            temp = array(ds, "temperature").astype(np.float64)
            speed = array(ds, "windSpeed").astype(np.float64)
            direction = array(ds, "windDir").astype(np.float64)
            altitude = array(ds, "altitude").astype(np.float64)
            lat = array(ds, "latitude").astype(np.float64)
            lon = lon_to_180(array(ds, "longitude").astype(np.float64))
            time_obs = array(ds, "timeObs").astype(np.float64)
            sources = array(ds, "dataSource", fill=-999).astype(np.int32)
            pressure = pressure_hpa_from_altitude_m(altitude)
            conus = in_strict_conus(lat, lon)

            qcr_arrays = {name: qcr(ds, name, temp.shape) for name in ABO_QCR_FIELDS}
            loc_ok = (
                np.isfinite(lat) & np.isfinite(lon) & np.isfinite(pressure)
                & (qcr_arrays["latitudeQCR"] == 0)
                & (qcr_arrays["longitudeQCR"] == 0)
                & (qcr_arrays["altitudeQCR"] == 0)
                & (pressure > 100.0) & (pressure < 1050.0)
            )
            temp_ok = loc_ok & (qcr_arrays["temperatureQCR"] == 0) & np.isfinite(temp) & (temp > 180.0) & (temp < 330.0)
            wind_ok = (
                loc_ok & (qcr_arrays["windSpeedQCR"] == 0) & (qcr_arrays["windDirQCR"] == 0)
                & np.isfinite(speed) & np.isfinite(direction)
                & (speed >= 0.0) & (speed < 150.0)
            )

            source_rows = []
            window_rows = []
            for source in sorted(np.unique(sources)):
                source_mask = sources == source
                source_rows.append({
                    "source_code": int(source),
                    "source_name": source_mapping(ds).get(int(source), "unknown"),
                    "n_raw": int(source_mask.sum()),
                    "n_strict_conus": int((source_mask & conus).sum()),
                    "n_outside_strict_conus": int((source_mask & ~conus).sum()),
                    "n_location_qc_pass": int((source_mask & loc_ok).sum()),
                    "n_temperature_qc_physical_pass": int((source_mask & temp_ok).sum()),
                    "n_wind_qc_physical_pass": int((source_mask & wind_ok).sum()),
                })
                for window, (lo, hi) in PRESSURE_WINDOWS.items():
                    level_mask = np.isfinite(pressure) & (pressure >= lo) & (pressure <= hi)
                    for variable, valid in (("temperature", temp_ok), ("wind", wind_ok)):
                        window_rows.append({
                            "source_code": int(source),
                            "source_name": source_mapping(ds).get(int(source), "unknown"),
                            "window": window,
                            "variable_group": variable,
                            "n_global": int((source_mask & valid & level_mask).sum()),
                            "n_strict_conus": int((source_mask & valid & level_mask & conus).sum()),
                        })

            finite_time = np.isfinite(time_obs)
            offsets = (time_obs[finite_time] - nominal_epoch) / 60.0
            file_row = {
                "n_records": int(temp.size),
                "time_finite_fraction": float(finite_time.mean()),
                "time_offset_min_minutes": float(np.min(offsets)),
                "time_offset_max_minutes": float(np.max(offsets)),
                "n_location_qc_pass": int(loc_ok.sum()),
                "n_temperature_qc_physical_pass": int(temp_ok.sum()),
                "n_wind_qc_physical_pass": int(wind_ok.sum()),
                "n_strict_conus_raw": int(conus.sum()),
                "qcr_nonzero_counts": json.dumps({name: int(np.sum(values != 0)) for name, values in qcr_arrays.items()}, sort_keys=True),
            }
            return file_row, source_rows, window_rows, source_mapping(ds)


def inspect_metar(path: Path, nominal_epoch: float):
    compressed = path.read_bytes()
    with tempfile.NamedTemporaryFile(suffix=".nc") as handle:
        handle.write(gzip.decompress(compressed))
        handle.flush()
        with netCDF4.Dataset(handle.name) as ds:
            temp = array(ds, "temperature").astype(np.float64)
            speed = array(ds, "windSpeed").astype(np.float64)
            direction = array(ds, "windDir").astype(np.float64)
            lat = array(ds, "latitude").astype(np.float64)
            lon = lon_to_180(array(ds, "longitude").astype(np.float64))
            time_obs = array(ds, "timeObs").astype(np.float64)
            names = decode_station_names(array(ds, "stationName", fill=b""))
            conus = in_strict_conus(lat, lon)
            loc_ok = np.isfinite(lat) & np.isfinite(lon)
            temp_qcr = qcr(ds, "temperatureQCR", temp.shape)
            speed_qcr = qcr(ds, "windSpeedQCR", temp.shape)
            direction_qcr = qcr(ds, "windDirQCR", temp.shape)
            temp_ok = loc_ok & (temp_qcr == 0) & np.isfinite(temp) & (temp > 180.0) & (temp < 330.0)
            wind_ok = (
                loc_ok & (speed_qcr == 0) & (direction_qcr == 0)
                & np.isfinite(speed) & np.isfinite(direction)
                & (speed >= 0.0) & (speed < 100.0)
            )
            finite_time = np.isfinite(time_obs)
            offsets = (time_obs[finite_time] - nominal_epoch) / 60.0
            file_row = {
                "n_records": int(temp.size),
                "time_finite_fraction": float(finite_time.mean()),
                "time_offset_min_minutes": float(np.min(offsets)),
                "time_offset_max_minutes": float(np.max(offsets)),
                "n_location_finite": int(loc_ok.sum()),
                "n_temperature_qc_physical_pass": int(temp_ok.sum()),
                "n_wind_qc_physical_pass": int(wind_ok.sum()),
                "n_strict_conus_raw": int(conus.sum()),
                "n_unique_stations_global": int(len(set(names[loc_ok]))),
                "n_unique_stations_strict_conus": int(len(set(names[loc_ok & conus]))),
                "qcr_nonzero_counts": json.dumps({
                    "temperatureQCR": int(np.sum(temp_qcr != 0)),
                    "windSpeedQCR": int(np.sum(speed_qcr != 0)),
                    "windDirQCR": int(np.sum(direction_qcr != 0)),
                }, sort_keys=True),
            }
            variable_rows = []
            for variable, mask in (("t2m", temp_ok), ("u10", wind_ok), ("v10", wind_ok)):
                variable_rows.append({
                    "variable": variable,
                    "n_global": int(mask.sum()),
                    "n_strict_conus": int((mask & conus).sum()),
                    "n_outside_strict_conus": int((mask & ~conus).sum()),
                })
            return file_row, variable_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames = list(rows[0])
    seen = set(fieldnames)
    for row in rows[1:]:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data/observation_interface_2019/raw/madis_stratified24_v1",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=ROOT / "reports/2026/07212026report/20260721__2019_independent_year_protocol_audit",
    )
    args = parser.parse_args()
    table_root = args.report_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)

    file_rows = []
    source_rows = []
    window_rows = []
    metar_rows = []
    mappings = []
    for dt in stratified24(args.year):
        nominal_epoch = dt.replace(tzinfo=timezone.utc).timestamp()
        for modality, product in (("ABO", "acars"), ("METAR", "metar")):
            path = (
                args.raw_root / "madisPublic1/data/archive"
                / f"{dt:%Y/%m/%d}/point/{product}/netcdf/{dt:%Y%m%d_%H%M}.gz"
            )
            common = {
                "datetime_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "modality": modality,
                "local_path": str(path),
            }
            if modality == "ABO":
                file_row, source_part, window_part, mapping = inspect_abo(path, nominal_epoch)
                mappings.append(mapping)
                source_rows.extend([{**common, **row} for row in source_part])
                window_rows.extend([{**common, **row} for row in window_part])
            else:
                file_row, variable_part = inspect_metar(path, nominal_epoch)
                metar_rows.extend([{**common, **row} for row in variable_part])
            file_rows.append({**common, **file_row})
            print(f"audited {modality:5s} {dt:%Y-%m-%d %H:%M}")

    unique_mappings = {json.dumps(item, sort_keys=True) for item in mappings}
    if len(unique_mappings) != 1:
        raise RuntimeError("MADIS ABO dataSource mapping changed across the 24 files")
    mapping = mappings[0]
    write_csv(table_root / "madis_2019_strat24_file_audit.csv", file_rows)
    write_csv(table_root / "abo_2019_strat24_source_qc_coverage.csv", source_rows)
    write_csv(table_root / "abo_2019_strat24_pressure_window_counts.csv", window_rows)
    write_csv(table_root / "metar_2019_strat24_variable_qc_coverage.csv", metar_rows)

    abo_sources = Counter()
    for row in source_rows:
        abo_sources[int(row["source_code"])] += int(row["n_raw"])
    summary = {
        "year": args.year,
        "n_files": len(file_rows),
        "n_abo_files": sum(row["modality"] == "ABO" for row in file_rows),
        "n_metar_files": sum(row["modality"] == "METAR" for row in file_rows),
        "all_time_obs_finite": all(math.isclose(float(row["time_finite_fraction"]), 1.0) for row in file_rows),
        "abo_data_source_mapping": mapping,
        "abo_raw_counts_by_source": dict(sorted(abo_sources.items())),
        "strict_conus_bounds": STRICT_CONUS,
        "pressure_windows_hpa": PRESSURE_WINDOWS,
    }
    (args.report_root / "madis_2019_strat24_full_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
