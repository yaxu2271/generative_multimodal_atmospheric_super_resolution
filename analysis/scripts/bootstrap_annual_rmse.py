#!/usr/bin/env python3
"""Moving-block bootstrap for the selected R+A+S interface in 2020.

The observation interfaces were selected using 2019 development data and are
evaluated on the full set of 723 matched 2020 analysis times. The bootstrap
operates on the full 2020 12-hour calendar: 732 slots, 723 matched effects, and
9 baseline-unavailable gaps.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_paths import output_path, required_path

REPORT_DIR = output_path("ANNUAL_RMSE_OUTPUT_ROOT", "annual_rmse")
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR = REPORT_DIR / "figures"
SOURCE_CSV = required_path("ANNUAL_RMSE_TIMESTEP_METRICS_CSV")

SEED = 20260726
BOOTSTRAP_REPLICATES = 10_000
BLOCK_DAYS_TO_CASES = {3: 6, 7: 14, 14: 28, 21: 42}
CALENDAR_TIMESTEPS = list(range(0, 1464, 2))

PROTOCOL = "RplusAplusS_selected_2019"
PROTOCOL_LABEL = "R+A+S (selected with 2019 data)"
REGIONS = ["strict_conus", "global", "global_excluding_strict_conus"]
REGION_LABELS = {
    "strict_conus": "Within CONUS",
    "global": "Global",
    "global_excluding_strict_conus": "Outside CONUS",
}
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
SHORT = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "geopotential_500": "z500",
    "geopotential_850": "z850",
    "u_component_of_wind_500": "u500",
    "u_component_of_wind_850": "u850",
    "v_component_of_wind_500": "v500",
    "v_component_of_wind_850": "v850",
    "temperature_500": "t500",
    "temperature_850": "t850",
    "specific_humidity_500": "q500",
    "specific_humidity_850": "q850",
}
SURFACE3 = VARIABLES[:3]
AIRCRAFT6 = [
    "u_component_of_wind_500",
    "u_component_of_wind_850",
    "v_component_of_wind_500",
    "v_component_of_wind_850",
    "temperature_500",
    "temperature_850",
]
VARIABLE_SETS = {
    "all13": VARIABLES,
    "surface3": SURFACE3,
    "aircraft6": AIRCRAFT6,
}
VARIABLE_SET_LABELS = {
    "all13": "All 13 variables",
    "surface3": "Surface-targeted",
    "aircraft6": "Aircraft-targeted",
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_source() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV)
    if "rmse_change_pct_vs_R" not in df.columns:
        if "pct_delta_vs_igra" not in df.columns:
            raise ValueError("Missing paired analysis-time RMSE-change column")
        df = df.rename(columns={"pct_delta_vs_igra": "rmse_change_pct_vs_R"})
    if df["protocol"].nunique() != 1 or df["protocol"].iloc[0] != PROTOCOL:
        raise ValueError("Unexpected protocol set in source table")
    if df["timestep"].nunique() != 723:
        raise ValueError(f"Expected 723 timesteps, found {df['timestep'].nunique()}")
    expected_rows = 1 * 3 * 13 * 723
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(df)}")
    if df.duplicated(["region", "variable", "timestep"]).any():
        raise ValueError("Duplicate region/variable/timestep keys")
    if not np.isfinite(df[["baseline_rmse", "method_rmse", "rmse_change_pct_vs_R"]].to_numpy()).all():
        raise ValueError("Non-finite metric values")
    return df


def build_calendar(valid_timesteps: list[int]) -> pd.DataFrame:
    valid = set(valid_timesteps)
    calendar = pd.DataFrame(
        {
            "slot_index": np.arange(len(CALENDAR_TIMESTEPS), dtype=int),
            "timestep": CALENDAR_TIMESTEPS,
        }
    )
    calendar["datetime_utc"] = pd.Timestamp("2020-01-01T00:00:00Z") + pd.to_timedelta(
        calendar["slot_index"] * 12, unit="h"
    )
    calendar["effect_available"] = calendar["timestep"].isin(valid)
    calendar["slot_status"] = np.where(
        calendar["effect_available"], "effect_available", "baseline_unavailable"
    )
    if int(calendar["effect_available"].sum()) != 723:
        raise ValueError("Calendar availability does not match the 723 evaluation times")
    calendar.to_csv(TABLE_DIR / "calendar_grid_2020_12h.csv", index=False)
    return calendar


def build_effect_series(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        region_df = df[df["region"].eq(region)]
        for varset, variables in VARIABLE_SETS.items():
            sub = region_df[region_df["variable"].isin(variables)]
            grouped = (
                sub.groupby(["region", "timestep", "datetime_utc", "month"], as_index=False)
                .agg(
                    effect_pct=("rmse_change_pct_vs_R", "mean"),
                    baseline_rmse=("baseline_rmse", "mean"),
                    method_rmse=("method_rmse", "mean"),
                )
            )
            grouped["series_id"] = f"aggregate__{region}__{varset}"
            grouped["analysis_scope"] = "aggregate_set"
            grouped["variable_set"] = varset
            grouped["variable_set_label"] = VARIABLE_SET_LABELS[varset]
            rows.append(grouped)
        for variable in VARIABLES:
            sub = region_df[region_df["variable"].eq(variable)].copy()
            grouped = sub[["region", "timestep", "datetime_utc", "month", "rmse_change_pct_vs_R", "baseline_rmse", "method_rmse"]].rename(
                columns={"rmse_change_pct_vs_R": "effect_pct"}
            )
            grouped["series_id"] = f"per_variable__{region}__{SHORT[variable]}"
            grouped["analysis_scope"] = "per_variable"
            grouped["variable"] = variable
            grouped["var_short"] = SHORT[variable]
            grouped["variable_set"] = "per_variable"
            grouped["variable_set_label"] = SHORT[variable]
            rows.append(grouped)
    series = pd.concat(rows, ignore_index=True)
    series["protocol"] = PROTOCOL
    series["display"] = PROTOCOL_LABEL
    series.to_csv(TABLE_DIR / "paired_effect_timeseries.csv", index=False)
    return series


def lag1_exact_12h(timesteps: np.ndarray, values: np.ndarray) -> tuple[float, int]:
    lookup = {int(t): float(v) for t, v in zip(timesteps, values)}
    xs, ys = [], []
    for t in timesteps:
        if int(t + 2) in lookup:
            xs.append(lookup[int(t)])
            ys.append(lookup[int(t + 2)])
    if len(xs) < 3:
        return float("nan"), len(xs)
    return float(np.corrcoef(xs, ys)[0, 1]), len(xs)


def ar1_neff(n: int, rho: float) -> tuple[float, float]:
    if not np.isfinite(rho):
        return float("nan"), float("nan")
    raw = n * (1.0 - rho) / (1.0 + rho) if rho < 0.999 else 1.0
    return float(raw), float(min(n, max(1.0, raw)))


def bootstrap_matrix(
    matrix: np.ndarray,
    retained_slots: np.ndarray,
    valid_mask: np.ndarray,
    block_cases: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_series, n_cases = matrix.shape
    n_slots = len(valid_mask)
    calendar_values = np.full((n_series, n_slots), np.nan, dtype=np.float64)
    calendar_values[:, retained_slots] = matrix

    starts = rng.integers(0, n_slots, size=(BOOTSTRAP_REPLICATES, math_blocks(n_cases, block_cases, valid_mask)), endpoint=False)
    means = np.empty((BOOTSTRAP_REPLICATES, n_series), dtype=np.float64)
    start_counts = np.zeros(n_slots, dtype=np.int64)
    effect_counts = np.zeros(n_slots, dtype=np.int64)
    for b in range(BOOTSTRAP_REPLICATES):
        collected_slots = []
        for start in starts[b]:
            start_counts[start] += 1
            slots = (start + np.arange(block_cases)) % n_slots
            slots = slots[valid_mask[slots]]
            if len(slots):
                collected_slots.extend(slots.tolist())
                for slot in slots:
                    effect_counts[slot] += 1
            if len(collected_slots) >= n_cases:
                break
        if len(collected_slots) < n_cases:
            # Extremely unlikely with the conservative block count, but keep deterministic fallback.
            while len(collected_slots) < n_cases:
                start = int(rng.integers(0, n_slots))
                start_counts[start] += 1
                slots = (start + np.arange(block_cases)) % n_slots
                slots = slots[valid_mask[slots]]
                collected_slots.extend(slots.tolist())
                for slot in slots:
                    effect_counts[slot] += 1
        selected = np.asarray(collected_slots[:n_cases], dtype=np.int64)
        means[b] = np.nanmean(calendar_values[:, selected], axis=1)
    return means, start_counts, effect_counts


def math_blocks(n_cases: int, block_cases: int, valid_mask: np.ndarray) -> int:
    valid_fraction = float(np.mean(valid_mask))
    expected_valid = max(1.0, block_cases * valid_fraction)
    return int(np.ceil(n_cases / expected_valid)) + 8


def summarize_effects(series: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[dict]]:
    rows = []
    matrices = []
    metadata = []
    for series_id, group in series.groupby("series_id", sort=False):
        group = group.sort_values("timestep")
        effects = group["effect_pct"].to_numpy(dtype=np.float64)
        timesteps = group["timestep"].to_numpy(dtype=np.int64)
        if len(effects) != 723:
            raise ValueError(f"{series_id} has {len(effects)} effects")
        rho1, pair_count = lag1_exact_12h(timesteps, effects)
        neff_raw, neff_capped = ar1_neff(len(effects), rho1)
        first = group.iloc[0]
        row = {
            "series_id": series_id,
            "analysis_scope": first["analysis_scope"],
            "protocol": PROTOCOL,
            "display": PROTOCOL_LABEL,
            "region": first["region"],
            "region_display": REGION_LABELS[first["region"]],
            "variable_set": first.get("variable_set", ""),
            "variable_set_label": first.get("variable_set_label", ""),
            "variable": first.get("variable", ""),
            "var_short": first.get("var_short", ""),
            "n_evaluation_times": len(effects),
            "mean_effect_pct": float(np.mean(effects)),
            "median_effect_pct": float(np.median(effects)),
            "fraction_analysis_times_improved": float(np.mean(effects < 0.0)),
            "lag1_autocorrelation": rho1,
            "lag1_exact_12h_pair_count": pair_count,
            "ar1_n_eff_raw": neff_raw,
            "ar1_n_eff_capped": neff_capped,
        }
        rows.append(row)
        matrices.append(effects)
        metadata.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "paired_effect_summary.csv", index=False)
    matrix = np.vstack(matrices)
    return summary, matrix, np.array(sorted(series["timestep"].unique()), dtype=np.int64), metadata


def run_bootstrap(effect_summary: pd.DataFrame, matrix: np.ndarray, timesteps: np.ndarray, calendar: pd.DataFrame) -> pd.DataFrame:
    slot_lookup = {int(t): int(s) for s, t in zip(calendar["slot_index"], calendar["timestep"])}
    retained_slots = np.array([slot_lookup[int(t)] for t in timesteps], dtype=np.int64)
    valid_mask = calendar["effect_available"].to_numpy(dtype=bool)
    rng_master = np.random.default_rng(SEED)
    rows = []
    replicate_payload = {}
    for block_days, block_cases in BLOCK_DAYS_TO_CASES.items():
        rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
        means, start_counts, effect_counts = bootstrap_matrix(matrix, retained_slots, valid_mask, block_cases, rng)
        replicate_payload[f"block_{block_days}day"] = means
        sample_mean = effect_summary["mean_effect_pct"].to_numpy(dtype=np.float64)
        boot_center = means.mean(axis=0)
        ci_low = np.percentile(means, 2.5, axis=0)
        ci_high = np.percentile(means, 97.5, axis=0)
        prob_lt0 = np.mean(means < 0.0, axis=0)
        for i, base in effect_summary.reset_index(drop=True).iterrows():
            rows.append(
                {
                    **base.to_dict(),
                    "block_days": block_days,
                    "block_cases": block_cases,
                    "calendar_slots": len(calendar),
                    "calendar_candidate_start_slots": len(calendar),
                    "calendar_edge_handling": "circular_wrap",
                    "baseline_unavailable_calendar_slots": int((~valid_mask).sum()),
                    "calibration_excluded_calendar_slots": 0,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "random_seed": SEED,
                    "bootstrap_mean_effect_pct": float(boot_center[i]),
                    "bootstrap_mean_bias_pct": float(boot_center[i] - sample_mean[i]),
                    "bootstrap_se_pct": float(np.std(means[:, i], ddof=1)),
                    "ci95_percentile_low_pct": float(ci_low[i]),
                    "ci95_percentile_high_pct": float(ci_high[i]),
                    "ci95_percentile_width_pct": float(ci_high[i] - ci_low[i]),
                    "ci95_primary_low_pct": float(ci_low[i]),
                    "ci95_primary_high_pct": float(ci_high[i]),
                    "ci95_primary_method": "raw_percentile",
                    "probability_effect_lt_zero": float(prob_lt0[i]),
                    "block_bootstrap_interpretation": (
                        "improvement_supported"
                        if ci_high[i] < 0.0
                        else "worsening_supported"
                        if ci_low[i] > 0.0
                        else "interval_includes_zero"
                    ),
                }
            )
        pd.DataFrame(
            {
                "slot_index": np.arange(len(start_counts), dtype=int),
                "timestep": calendar["timestep"],
                "block_days": block_days,
                "start_count": start_counts,
                "effect_count": effect_counts,
            }
        ).to_csv(TABLE_DIR / f"calendar_block_coverage_{block_days}day.csv", index=False)
    np.savez_compressed(
        TABLE_DIR / "bootstrap_mean_replicates.npz",
        **replicate_payload,
        series_id=effect_summary["series_id"].to_numpy(dtype=str),
    )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "area_weighted_rmse_block_sensitivity.csv", index=False)
    primary = out[out["block_days"].eq(14)].copy()
    primary.to_csv(TABLE_DIR / "area_weighted_rmse_14day_intervals.csv", index=False)
    return out


def plot_aggregate(primary: pd.DataFrame) -> None:
    data = primary[primary["analysis_scope"].eq("aggregate_set")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.0), sharey=True, constrained_layout=True)
    marker = {"all13": "o", "surface3": "s", "aircraft6": "D"}
    colors = {"all13": "#4A5568", "surface3": "#0B7A75", "aircraft6": "#6F4AA8"}
    for ax, region in zip(axes, REGIONS):
        sub = data[data["region"].eq(region)].copy()
        y = np.arange(len(VARIABLE_SETS))
        for j, varset in enumerate(VARIABLE_SETS):
            row = sub[sub["variable_set"].eq(varset)].iloc[0]
            ax.errorbar(
                row["mean_effect_pct"],
                j,
                xerr=[
                    [row["mean_effect_pct"] - row["ci95_primary_low_pct"]],
                    [row["ci95_primary_high_pct"] - row["mean_effect_pct"]],
                ],
                fmt=marker[varset],
                markersize=7,
                markerfacecolor="white",
                markeredgewidth=1.7,
                color=colors[varset],
                linewidth=1.9,
            )
        ax.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--")
        ax.set_title(REGION_LABELS[region], weight="bold", fontsize=14)
        ax.set_xlabel("Mean RMSE change and 95% CI (%)")
        ax.grid(axis="x", color="#d8dee9", alpha=0.85)
    axes[0].set_yticks(np.arange(len(VARIABLE_SETS)))
    axes[0].set_yticklabels([VARIABLE_SET_LABELS[k] for k in VARIABLE_SETS], fontsize=12)
    fig.suptitle("R+A+S selected with 2019 data: 14-day moving-block bootstrap over 2020", weight="bold", fontsize=16)
    fig.savefig(FIG_DIR / "selected_RAS_14day_block_bootstrap_aggregate.png", dpi=260)
    fig.savefig(FIG_DIR / "selected_RAS_14day_block_bootstrap_aggregate.pdf")
    plt.close(fig)


def plot_per_variable(primary: pd.DataFrame) -> None:
    data = primary[
        (primary["analysis_scope"].eq("per_variable"))
        & (primary["region"].eq("strict_conus"))
    ].copy()
    order = [SHORT[v] for v in VARIABLES]
    data["order"] = data["var_short"].map({v: i for i, v in enumerate(order)})
    data = data.sort_values("order")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8.2, 7.8), constrained_layout=True)
    ax.errorbar(
        data["mean_effect_pct"],
        y,
        xerr=[
            data["mean_effect_pct"] - data["ci95_primary_low_pct"],
            data["ci95_primary_high_pct"] - data["mean_effect_pct"],
        ],
        fmt="D",
        markersize=5.2,
        markerfacecolor="white",
        markeredgewidth=1.5,
        color="#0B7A75",
        linewidth=1.6,
    )
    ax.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(data["var_short"], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Mean RMSE change and 95% CI (%)")
    ax.set_title("Strict CONUS per-variable 14-day block robustness", weight="bold")
    ax.grid(axis="x", color="#d8dee9", alpha=0.85)
    fig.savefig(FIG_DIR / "selected_RAS_14day_block_bootstrap_conus_per_variable.png", dpi=260)
    fig.savefig(FIG_DIR / "selected_RAS_14day_block_bootstrap_conus_per_variable.pdf")
    plt.close(fig)


def write_readme(bootstrap: pd.DataFrame) -> None:
    primary = bootstrap[bootstrap["block_days"].eq(14)].copy()
    agg = primary[primary["analysis_scope"].eq("aggregate_set")]
    strict = agg[agg["region"].eq("strict_conus")]
    lines = [
        "# R+A+S Moving-Block Bootstrap",
        "",
        "This report evaluates serial-correlation robustness for the R+A+S interface selected with 2019 development data and applied to 2020.",
        "",
        "This uses all 723 matched 2020 cases because the observation interfaces were selected with 2019 development data. The 2020 calendar has 732 12-hour slots; the 9 unmatched slots remain explicit gaps.",
        "",
        "## Main 14-day results",
        "",
    ]
    for varset in VARIABLE_SETS:
        row = strict[strict["variable_set"].eq(varset)].iloc[0]
        lines.append(
            f"- Strict CONUS {VARIABLE_SET_LABELS[varset]}: {row['mean_effect_pct']:+.3f}% "
            f"[{row['ci95_primary_low_pct']:+.3f}, {row['ci95_primary_high_pct']:+.3f}], "
            f"P(effect < 0)={row['probability_effect_lt_zero']:.4f}."
        )
    rest = agg[(agg["region"].eq("global_excluding_strict_conus")) & (agg["variable_set"].eq("all13"))].iloc[0]
    lines.extend(
        [
            "",
            f"- Outside CONUS all13: {rest['mean_effect_pct']:+.4f}% "
            f"[{rest['ci95_primary_low_pct']:+.4f}, {rest['ci95_primary_high_pct']:+.4f}].",
            "",
            "## Outputs",
            "",
            f"- Full bootstrap summary: `{TABLE_DIR / 'area_weighted_rmse_block_sensitivity.csv'}`",
            f"- 14-day aggregate and per-variable table: `{TABLE_DIR / 'area_weighted_rmse_14day_intervals.csv'}`",
            f"- Figures: `{FIG_DIR}`",
            "",
        ]
    )
    (REPORT_DIR / "README_moving_block_bootstrap.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = read_source()
    calendar = build_calendar(sorted(df["timestep"].unique()))
    series = build_effect_series(df)
    effect_summary, matrix, timesteps, _metadata = summarize_effects(series)
    bootstrap = run_bootstrap(effect_summary, matrix, timesteps, calendar)
    primary = bootstrap[bootstrap["block_days"].eq(14)].copy()
    plot_aggregate(primary)
    plot_per_variable(primary)
    write_readme(bootstrap)
    print(
        primary[
            (primary["analysis_scope"].eq("aggregate_set"))
            & (primary["region"].eq("strict_conus"))
        ][
            [
                "variable_set",
                "mean_effect_pct",
                "ci95_primary_low_pct",
                "ci95_primary_high_pct",
                "probability_effect_lt_zero",
                "block_bootstrap_interpretation",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
