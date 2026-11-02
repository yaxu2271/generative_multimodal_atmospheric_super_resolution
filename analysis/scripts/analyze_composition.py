#!/usr/bin/env python3
"""Summarize area-weighted R+A, R+S, and R+A+S annual RMSE changes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area-weighted-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.area_weighted_summary)
    required = {
        "protocol", "region", "variable_group", "n_timesteps",
        "area_weighted_mean_pct_delta",
    }
    if not required.issubset(source.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(source.columns))}")
    summary = source.rename(
        columns={"protocol": "configuration", "n_timesteps": "n_analysis_times"}
    )
    if set(summary["n_analysis_times"]) != {723}:
        raise ValueError("All composition summaries must use 723 analysis times")
    summary.to_csv(args.output_dir / "area_weighted_composition_summary.csv", index=False)

    within_conus = summary[summary["region"].eq("strict_conus")]
    rows = []
    labels = {"all13": "all13", "surface3": "surface-targeted", "aircraft6": "aircraft-targeted"}
    for variable_group, label in labels.items():
        values = within_conus[within_conus["variable_group"].eq(variable_group)].set_index(
            "configuration"
        )["area_weighted_mean_pct_delta"]
        best_single = min(values["R+A"], values["R+S"])
        rows.append(
            {
                "metric": label,
                "R+A": values["R+A"],
                "R+S": values["R+S"],
                "R+A+S": values["R+A+S"],
                "R+A+S_minus_best_single_pp": values["R+A+S"] - best_single,
                "best_single": "R+A" if values["R+A"] <= values["R+S"] else "R+S",
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "area_weighted_complementarity_summary.csv", index=False)


if __name__ == "__main__":
    main()
