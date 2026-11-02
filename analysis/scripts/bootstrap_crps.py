#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis_paths import output_path, required_path

ROOT = output_path("PROBABILISTIC_ANALYSIS_OUTPUT_ROOT", "probabilistic_metrics")
PAIRED = required_path("PAIRED_CRPS_METRICS_CSV")
OUT = ROOT / "tables/area_weighted_crps_block_sensitivity.csv"
GROUPS = {
    "all_13_variables": lambda frame: np.ones(len(frame), dtype=bool),
    "surface_targeted_variables": lambda frame: frame["is_surface_targeted"].to_numpy(bool),
    "aircraft_targeted_variables": lambda frame: frame["is_aircraft_targeted"].to_numpy(bool),
}
CALENDAR_SIZE = 366 * 2
BLOCK_DAYS = (3, 7, 14)
REPLICATES = 10_000
SEED = 20260731


def block_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    block_length: int,
) -> tuple[float, float]:
    blocks = int(np.ceil(CALENDAR_SIZE / block_length))
    offsets = np.arange(block_length)
    means = np.empty(REPLICATES, dtype=float)
    for start in range(0, REPLICATES, 500):
        stop = min(start + 500, REPLICATES)
        starts = rng.integers(0, CALENDAR_SIZE, size=(stop - start, blocks, 1))
        indices = ((starts + offsets) % CALENDAR_SIZE).reshape(stop - start, -1)[:, :CALENDAR_SIZE]
        means[start:stop] = np.nanmean(values[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975], method="linear")
    return float(low), float(high)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PAIRED)
    if frame["timestep"].mod(2).any():
        raise ValueError("Expected the 00/12 UTC evaluation calendar to use even six-hour indices")
    rng = np.random.default_rng(SEED)
    rows = []
    for region in ["strict_conus", "global", "outside_conus"]:
        regional = frame[frame["region"].eq(region)]
        for group, selector in GROUPS.items():
            subset = regional[selector(regional)]
            by_time = subset.groupby("timestep")["crps_pct_change"].mean()
            calendar = np.full(CALENDAR_SIZE, np.nan, dtype=float)
            calendar[(by_time.index.to_numpy(int) // 2)] = by_time.to_numpy(float)
            for block_days in BLOCK_DAYS:
                block_length = block_days * 2
                low, high = block_interval(calendar, rng, block_length)
                rows.append({
                    "region": region,
                    "evaluation_group": group,
                    "n_observed_cases": int(np.isfinite(calendar).sum()),
                    "n_calendar_holes": int(np.isnan(calendar).sum()),
                    "block_length_days": block_days,
                    "block_length_cases": block_length,
                    "replicates": REPLICATES,
                    "seed": SEED,
                    "mean_crps_pct_change": float(np.nanmean(calendar)),
                    "ci95_low": low,
                    "ci95_high": high,
                })
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
