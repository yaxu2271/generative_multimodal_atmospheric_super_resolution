#!/usr/bin/env python3
"""Build DJ-compatible 13-channel IGRA observations from NOAA IGRA v2 raw files.

The output follows the nested pickle layout used by DJ's ``igra_2020_all.pkl``:
``all_times[t] == [[query_locations_by_channel], [values_by_channel]]``.
Only requested nominal sounding times are populated; the remaining six-hourly
slots are empty so the timestep index stays aligned with the ERA5 year split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np

from independent_year_common import ROOT, six_hour_index, stratified24


VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "geopotential_500",
    "geopotential_850",
    "u_component_of_wind_500",
    "u_component_of_wind_850",
    "v_component_of_wind_500",
    "v_component_of_wind_850",
    "temperature_500",
    "temperature_850",
    "specific_humidity_500",
    "specific_humidity_850",
]


def _integer(text: str) -> int:
    text = text.strip()
    return int(text) if text else -9999


def _valid(value: int) -> bool:
    return value not in {-8888, -9999}


def _specific_humidity(temp_c: float, dpdp_c: float, pressure_hpa: float) -> float:
    dewpoint_c = temp_c - dpdp_c
    vapor_pressure_hpa = 6.112 * math.exp(17.67 * dewpoint_c / (dewpoint_c + 243.5))
    return 0.622 * vapor_pressure_hpa / (pressure_hpa - 0.378 * vapor_pressure_hpa)


def _wind_components(direction_deg: float, speed_ms: float) -> tuple[float, float]:
    angle = math.radians(direction_deg)
    return -speed_ms * math.sin(angle), -speed_ms * math.cos(angle)


def _parse_level(line: str) -> dict[str, int | str]:
    return {
        "major": line[0:1],
        "minor": line[1:2],
        "pressure_pa": _integer(line[9:15]),
        "gph_m": _integer(line[16:21]),
        "temp_tenths_c": _integer(line[22:27]),
        "rh_tenths_pct": _integer(line[28:33]),
        "dpdp_tenths_c": _integer(line[34:39]),
        "wind_dir_deg": _integer(line[40:45]),
        "wind_speed_tenths_ms": _integer(line[46:51]),
    }


def _first_valid(levels: Iterable[dict], field: str):
    for level in levels:
        value = int(level[field])
        if _valid(value):
            return value
    return None


def _profile_values(levels: list[dict]) -> list[float | None]:
    surface = [level for level in levels if level["minor"] == "1"]
    p500 = [level for level in levels if level["pressure_pa"] == 50000]
    p850 = [level for level in levels if level["pressure_pa"] == 85000]

    result: list[float | None] = [None] * len(VARIABLES)
    temp = _first_valid(surface, "temp_tenths_c")
    if temp is not None:
        result[0] = temp / 10.0 + 273.15

    direction = _first_valid(surface, "wind_dir_deg")
    speed = _first_valid(surface, "wind_speed_tenths_ms")
    if direction is not None and speed is not None:
        result[1], result[2] = _wind_components(direction, speed / 10.0)

    for level_hpa, rows, gph_idx, u_idx, v_idx, temp_idx, q_idx in [
        (500, p500, 3, 5, 7, 9, 11),
        (850, p850, 4, 6, 8, 10, 12),
    ]:
        gph = _first_valid(rows, "gph_m")
        if gph is not None:
            result[gph_idx] = gph * 9.8

        direction = _first_valid(rows, "wind_dir_deg")
        speed = _first_valid(rows, "wind_speed_tenths_ms")
        if direction is not None and speed is not None:
            result[u_idx], result[v_idx] = _wind_components(direction, speed / 10.0)

        temp = _first_valid(rows, "temp_tenths_c")
        if temp is not None:
            result[temp_idx] = temp / 10.0 + 273.15
        dpdp = _first_valid(rows, "dpdp_tenths_c")
        if temp is not None and dpdp is not None:
            result[q_idx] = _specific_humidity(temp / 10.0, dpdp / 10.0, level_hpa)
    return result


def _iter_profiles(path: Path, targets: set[datetime]):
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise ValueError(f"Expected one member in {path}, found {members}")
        with archive.open(members[0], "r") as handle:
            while True:
                raw_header = handle.readline()
                if not raw_header:
                    break
                header = raw_header.decode("ascii").rstrip("\r\n")
                if not header.startswith("#"):
                    raise ValueError(f"Malformed IGRA header in {path}: {header[:80]!r}")
                n_levels = int(header[32:36])
                station_id = header[1:12]
                lat = int(header[55:62]) / 10000.0
                lon = int(header[63:71]) / 10000.0
                try:
                    dt = datetime(
                        int(header[13:17]),
                        int(header[18:20]),
                        int(header[21:23]),
                        int(header[24:26]),
                    )
                except ValueError:
                    # IGRA uses sentinel date/hour fields (for example hour=99)
                    # when nominal launch time is unknown. Such profiles cannot
                    # be matched to the prespecified 00 UTC development cases.
                    dt = None
                levels = []
                for _ in range(n_levels):
                    line = handle.readline().decode("ascii").rstrip("\r\n")
                    if dt in targets:
                        levels.append(_parse_level(line))
                if dt in targets:
                    yield station_id, dt, lat, lon, levels


def _year_length(year: int) -> int:
    return int((datetime(year + 1, 1, 1) - datetime(year, 1, 1)).total_seconds() // (6 * 3600))


def _empty_year(year: int):
    result = []
    for _ in range(_year_length(year)):
        locations = [np.empty((0, 2), dtype=np.float64) for _ in VARIABLES]
        values = [np.empty((0,), dtype=np.float64) for _ in VARIABLES]
        result.append([[locations], [values]])
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--station_zip_root", type=Path, required=True)
    parser.add_argument(
        "--normalization_root",
        type=Path,
        default=ROOT / "data_from_DJ_original_NERSC/1.40625deg_from_full_res_1_step_6hr_h5df",
    )
    parser.add_argument("--output_pkl", type=Path, required=True)
    parser.add_argument("--audit_csv", type=Path, required=True)
    parser.add_argument("--manifest_json", type=Path, required=True)
    parser.add_argument("--all_00_12", action="store_true")
    args = parser.parse_args()

    if args.all_00_12:
        targets = {
            datetime(args.year, 1, 1) + timedelta(hours=12 * i)
            for i in range(_year_length(args.year) // 2)
        }
    else:
        targets = set(stratified24(args.year))

    means_npz = np.load(args.normalization_root / "normalize_mean.npz")
    stds_npz = np.load(args.normalization_root / "normalize_std.npz")
    means = np.asarray([means_npz[name] for name in VARIABLES], dtype=np.float64).reshape(-1)
    stds = np.asarray([stds_npz[name] for name in VARIABLES], dtype=np.float64).reshape(-1)

    physical: dict[datetime, list[list[tuple[float, float, float, str]]]] = {
        dt: [[] for _ in VARIABLES] for dt in targets
    }
    duplicate_profiles = defaultdict(int)
    seen_profiles: set[tuple[str, datetime]] = set()
    zip_paths = sorted(args.station_zip_root.glob("*-data.txt.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No IGRA station ZIP files under {args.station_zip_root}")

    for path in zip_paths:
        for station_id, dt, lat, lon, levels in _iter_profiles(path, targets):
            key = (station_id, dt)
            duplicate_profiles[key] += int(key in seen_profiles)
            seen_profiles.add(key)
            for channel, value in enumerate(_profile_values(levels)):
                if value is not None and np.isfinite(value):
                    physical[dt][channel].append((lat, lon, float(value), station_id))

    output = _empty_year(args.year)
    audit_rows = []
    for dt in sorted(targets):
        index = six_hour_index(dt)
        for channel, variable in enumerate(VARIABLES):
            rows = physical[dt][channel]
            if rows:
                locations = np.asarray([[r[0], r[1]] for r in rows], dtype=np.float64)
                raw_values = np.asarray([r[2] for r in rows], dtype=np.float64)
                normalized = (raw_values - means[channel]) / stds[channel]
            else:
                locations = np.empty((0, 2), dtype=np.float64)
                raw_values = np.empty((0,), dtype=np.float64)
                normalized = np.empty((0,), dtype=np.float64)
            output[index][0][0][channel] = locations
            output[index][1][0][channel] = normalized
            audit_rows.append(
                {
                    "datetime_utc": dt.isoformat() + "Z",
                    "timestep_index": index,
                    "variable": variable,
                    "n_obs": len(rows),
                    "physical_min": float(raw_values.min()) if len(rows) else "",
                    "physical_max": float(raw_values.max()) if len(rows) else "",
                    "physical_mean": float(raw_values.mean()) if len(rows) else "",
                    "normalized_min": float(normalized.min()) if len(rows) else "",
                    "normalized_max": float(normalized.max()) if len(rows) else "",
                }
            )

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with args.audit_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    manifest = {
        "year": args.year,
        "target_datetimes": [dt.isoformat() + "Z" for dt in sorted(targets)],
        "variables": VARIABLES,
        "station_zip_root": str(args.station_zip_root.resolve()),
        "n_station_zips": len(zip_paths),
        "n_profiles_retained": len(seen_profiles),
        "n_duplicate_station_datetimes": int(sum(duplicate_profiles.values())),
        "normalization_root": str(args.normalization_root.resolve()),
        "normalization_mean_sha256": sha256(args.normalization_root / "normalize_mean.npz"),
        "normalization_std_sha256": sha256(args.normalization_root / "normalize_std.npz"),
        "output_pkl": str(args.output_pkl.resolve()),
        "output_pkl_sha256": sha256(args.output_pkl),
        "transform": {
            "temperature": "tenths C / 10 + 273.15 K",
            "geopotential": "GPH meters * 9.8",
            "wind": "u=-speed*sin(direction), v=-speed*cos(direction)",
            "specific_humidity": "q=0.622*e/(p-0.378*e), e=6.112*exp(17.67*Td/(Td+243.5))",
        },
    }
    args.manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
