#!/usr/bin/env python3
"""Compare a NOAA-raw IGRA rebuild with DJ's reference pickle by coordinate."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from build_igra_from_noaa_por import VARIABLES
from independent_year_common import six_hour_index, stratified24


def coordinate_values(locations, values):
    grouped = {}
    for (lat, lon), value in zip(np.asarray(locations), np.asarray(values)):
        grouped.setdefault((round(float(lat), 4), round(float(lon), 4)), []).append(float(value))
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_pkl", type=Path, required=True)
    parser.add_argument("--rebuilt_pkl", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    args = parser.parse_args()

    with args.reference_pkl.open("rb") as handle:
        reference = pickle.load(handle)
    with args.rebuilt_pkl.open("rb") as handle:
        rebuilt = pickle.load(handle)

    rows = []
    all_differences = []
    total_reference = total_rebuilt = total_overlap = 0
    for dt in stratified24(args.year):
        index = six_hour_index(dt)
        ref_locations, ref_values = reference[index][0][0], reference[index][1][0]
        new_locations, new_values = rebuilt[index][0][0], rebuilt[index][1][0]
        for channel, variable in enumerate(VARIABLES):
            ref = coordinate_values(ref_locations[channel], ref_values[channel])
            new = coordinate_values(new_locations[channel], new_values[channel])
            shared = sorted(set(ref) & set(new))
            differences = []
            duplicate_mismatch = 0
            for coordinate in shared:
                left = sorted(ref[coordinate])
                right = sorted(new[coordinate])
                count = min(len(left), len(right))
                duplicate_mismatch += abs(len(left) - len(right))
                differences.extend(abs(left[i] - right[i]) for i in range(count))
            total_reference += sum(len(v) for v in ref.values())
            total_rebuilt += sum(len(v) for v in new.values())
            total_overlap += len(differences)
            all_differences.extend(differences)
            rows.append(
                {
                    "datetime_utc": dt.isoformat() + "Z",
                    "variable": variable,
                    "n_reference": sum(len(v) for v in ref.values()),
                    "n_rebuilt": sum(len(v) for v in new.values()),
                    "n_coordinate_value_pairs_compared": len(differences),
                    "reference_pair_coverage": len(differences) / max(1, sum(len(v) for v in ref.values())),
                    "duplicate_count_mismatch": duplicate_mismatch,
                    "normalized_abs_diff_mean": float(np.mean(differences)) if differences else "",
                    "normalized_abs_diff_median": float(np.median(differences)) if differences else "",
                    "normalized_abs_diff_p95": float(np.quantile(differences, 0.95)) if differences else "",
                    "fraction_within_1e-6": float(np.mean(np.asarray(differences) <= 1e-6)) if differences else "",
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "reference_pkl": str(args.reference_pkl.resolve()),
        "rebuilt_pkl": str(args.rebuilt_pkl.resolve()),
        "year": args.year,
        "n_reference_values": total_reference,
        "n_rebuilt_values": total_rebuilt,
        "n_compared_values": total_overlap,
        "reference_pair_coverage": total_overlap / max(1, total_reference),
        "normalized_abs_diff_mean": float(np.mean(all_differences)) if all_differences else None,
        "normalized_abs_diff_median": float(np.median(all_differences)) if all_differences else None,
        "normalized_abs_diff_p95": float(np.quantile(all_differences, 0.95)) if all_differences else None,
        "fraction_within_1e-6": float(np.mean(np.asarray(all_differences) <= 1e-6)) if all_differences else None,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
