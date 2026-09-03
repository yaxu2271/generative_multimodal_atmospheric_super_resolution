#!/usr/bin/env python3
"""Apply the prespecified complete-archive IGRA parity gate."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from build_igra_from_noaa_por import VARIABLES
from independent_year_common import six_hour_index, stratified24


def check_pickle(path: Path, year: int) -> dict:
    expected_slots = 1464 if year % 4 == 0 else 1460
    with path.open("rb") as handle:
        data = pickle.load(handle)
    failures = []
    if len(data) != expected_slots:
        failures.append(f"expected {expected_slots} slots, found {len(data)}")
    counts = []
    for dt in stratified24(year):
        index = six_hour_index(dt)
        locations, values = data[index][0][0], data[index][1][0]
        if len(locations) != len(VARIABLES) or len(values) != len(VARIABLES):
            failures.append(f"{dt.isoformat()} does not contain 13 channels")
            continue
        count = 0
        for channel in range(len(VARIABLES)):
            locs = np.asarray(locations[channel])
            vals = np.asarray(values[channel])
            if locs.shape != (vals.size, 2):
                failures.append(f"{dt.isoformat()} channel {channel} shape mismatch")
            if not np.isfinite(locs).all() or not np.isfinite(vals).all():
                failures.append(f"{dt.isoformat()} channel {channel} has nonfinite values")
            count += vals.size
        if count == 0:
            failures.append(f"{dt.isoformat()} has no retained observations")
        counts.append(count)
    return {
        "path": str(path.resolve()),
        "year": year,
        "n_slots": len(data),
        "expected_slots": expected_slots,
        "stratified24_total_observations_min": min(counts) if counts else 0,
        "stratified24_total_observations_max": max(counts) if counts else 0,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-summary", type=Path, required=True)
    parser.add_argument("--rebuilt-2020", type=Path, required=True)
    parser.add_argument("--rebuilt-2019", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parity = json.loads(args.parity_summary.read_text())
    checks = {
        "reference_pair_coverage_ge_0p95": parity["reference_pair_coverage"] >= 0.95,
        "fraction_within_1e6_ge_0p99": parity["fraction_within_1e-6"] >= 0.99,
        "median_abs_diff_le_1e6": parity["normalized_abs_diff_median"] <= 1e-6,
        "p95_abs_diff_le_1e5": parity["normalized_abs_diff_p95"] <= 1e-5,
    }
    rebuilt_2020 = check_pickle(args.rebuilt_2020, 2020)
    rebuilt_2019 = check_pickle(args.rebuilt_2019, 2019)
    passed = all(checks.values()) and not rebuilt_2020["failures"] and not rebuilt_2019["failures"]
    result = {
        "status": "PASS" if passed else "FAIL",
        "prespecified_threshold_checks": checks,
        "parity_summary": parity,
        "rebuilt_2020_structure": rebuilt_2020,
        "rebuilt_2019_structure": rebuilt_2019,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
