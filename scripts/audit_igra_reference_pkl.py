#!/usr/bin/env python3
"""Export structural and numerical fingerprints of the 2020 IGRA reference pkl."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

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


def unpack_timestep(item):
    queries = item[0][0]
    values = item[1][0]
    if len(queries) != len(VARIABLES) or len(values) != len(VARIABLES):
        raise ValueError("Reference pkl does not contain 13 channels")
    return queries, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl",
        type=Path,
        default=ROOT / "original_DJ_selected_from_NERSC/IGRA_cond/samples/igra_2020_all.pkl",
    )
    parser.add_argument(
        "--normalization-root",
        type=Path,
        default=ROOT / "data_from_DJ_original_NERSC/1.40625deg_from_full_res_1_step_6hr_h5df",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=ROOT / "reports/2026/07212026report/20260721__2019_independent_year_protocol_audit",
    )
    parser.add_argument("--year", type=int, default=2020)
    args = parser.parse_args()

    with args.pkl.open("rb") as handle:
        reference = pickle.load(handle)
    means_archive = np.load(args.normalization_root / "normalize_mean.npz")
    stds_archive = np.load(args.normalization_root / "normalize_std.npz")
    means = np.asarray([float(np.asarray(means_archive[name]).reshape(-1)[0]) for name in VARIABLES])
    stds = np.asarray([float(np.asarray(stds_archive[name]).reshape(-1)[0]) for name in VARIABLES])

    summary_rows = []
    sample_rows = []
    indices = [six_hour_index(dt) for dt in stratified24(args.year)]
    for index in indices:
        queries, values = unpack_timestep(reference[index])
        for channel, name in enumerate(VARIABLES):
            locs = np.asarray(queries[channel], dtype=np.float64)
            normalized = np.asarray(values[channel], dtype=np.float64).reshape(-1)
            physical = normalized * stds[channel] + means[channel]
            if locs.shape != (normalized.size, 2):
                raise ValueError(f"Shape mismatch at t={index}, channel={name}")
            summary_rows.append({
                "timestep_index": index,
                "variable": name,
                "n_obs": normalized.size,
                "lat_min": float(np.min(locs[:, 0])) if normalized.size else np.nan,
                "lat_max": float(np.max(locs[:, 0])) if normalized.size else np.nan,
                "lon_min": float(np.min(locs[:, 1])) if normalized.size else np.nan,
                "lon_max": float(np.max(locs[:, 1])) if normalized.size else np.nan,
                "normalized_min": float(np.min(normalized)) if normalized.size else np.nan,
                "normalized_max": float(np.max(normalized)) if normalized.size else np.nan,
                "physical_min": float(np.min(physical)) if normalized.size else np.nan,
                "physical_max": float(np.max(physical)) if normalized.size else np.nan,
                "physical_mean": float(np.mean(physical)) if normalized.size else np.nan,
            })
            if index == indices[0]:
                for rank in range(min(12, normalized.size)):
                    sample_rows.append({
                        "timestep_index": index,
                        "variable": name,
                        "rank": rank,
                        "latitude": float(locs[rank, 0]),
                        "longitude": float(locs[rank, 1]),
                        "normalized_value": float(normalized[rank]),
                        "physical_value": float(physical[rank]),
                    })

    table_root = args.report_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        (f"igra_{args.year}_reference_strat24_channel_fingerprint.csv", summary_rows),
        (f"igra_{args.year}_reference_t0000_first12_records.csv", sample_rows),
    ):
        path = table_root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "reference_pkl": str(args.pkl),
        "calendar_year": args.year,
        "n_timesteps": len(reference),
        "n_channels": len(VARIABLES),
        "variables": VARIABLES,
        "normalization_mean": dict(zip(VARIABLES, means.tolist())),
        "normalization_std": dict(zip(VARIABLES, stds.tolist())),
        "fingerprinted_timesteps": indices,
    }
    (args.report_root / f"igra_{args.year}_reference_fingerprint_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
