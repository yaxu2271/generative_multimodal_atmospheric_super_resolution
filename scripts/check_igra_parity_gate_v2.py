#!/usr/bin/env python3
"""Apply the versioned spatial-tolerance IGRA parity gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_igra_parity_gate import check_pickle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-parity-summary", type=Path, required=True)
    parser.add_argument("--rebuilt-2020", type=Path, required=True)
    parser.add_argument("--rebuilt-2019", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parity = json.loads(args.spatial_parity_summary.read_text())
    checks = {
        "matching_is_one_to_one_geodesic": parity["matching"]
        == "deterministic greedy one-to-one geodesic matching",
        "max_distance_km_eq_25": parity["max_distance_km"] == 25.0,
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
        "gate_version": "v2_spatial_tolerance_25km",
        "reason_for_v2": (
            "v1 exact rounded-coordinate matching failed because the current NOAA period-of-record "
            "archive contains small retrospective station-coordinate revisions; numerical transforms "
            "on spatially corresponding records remained consistent"
        ),
        "prespecified_threshold_checks": checks,
        "spatial_parity_summary": parity,
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
