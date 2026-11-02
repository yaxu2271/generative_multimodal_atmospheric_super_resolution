#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_paths import output_path, required_path

ERA5_ROOT = required_path("ERA5_ROOT")
TIMESTEP_MANIFEST = required_path("EVALUATION_TIMESTEP_MANIFEST")
R_ROOT = required_path("R_ONLY_SAMPLES_ROOT")
RAS_ROOT = required_path("RAS_SAMPLES_ROOT")
OUT = output_path("PROBABILISTIC_ANALYSIS_OUTPUT_ROOT", "probabilistic_metrics")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

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
SURFACE = {"2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"}
AIRCRAFT = {
    "temperature_500",
    "temperature_850",
    "u_component_of_wind_500",
    "u_component_of_wind_850",
    "v_component_of_wind_500",
    "v_component_of_wind_850",
}
PROTOCOLS = {
    "R": R_ROOT,
    "R+A+S": RAS_ROOT,
}
COLORS = {"R": "#2B6EA6", "R+A+S": "#1B8E72"}


def load_timesteps() -> list[int]:
    payload = json.loads(TIMESTEP_MANIFEST.read_text())
    timesteps = [int(x) for x in payload["timesteps"]]
    if len(timesteps) != 723 or len(set(timesteps)) != 723:
        raise ValueError(f"Expected 723 unique timesteps, found {len(timesteps)}")
    return timesteps


def load_normalization() -> tuple[np.ndarray, np.ndarray]:
    means = np.load(ERA5_ROOT / "normalize_mean.npz")
    stds = np.load(ERA5_ROOT / "normalize_std.npz")
    mean = np.asarray([float(np.asarray(means[v]).reshape(-1)[0]) for v in VARIABLES], dtype=np.float32)
    std = np.asarray([float(np.asarray(stds[v]).reshape(-1)[0]) for v in VARIABLES], dtype=np.float32)
    return mean[:, None, None], std[:, None, None]


def load_truth(timestep: int) -> np.ndarray:
    with h5py.File(ERA5_ROOT / "test" / f"2020_{timestep:04d}.h5", "r") as handle:
        return np.stack(
            [np.asarray(handle["input"][variable], dtype=np.float32) for variable in VARIABLES],
            axis=0,
        )


def supports() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    lat = np.load(ERA5_ROOT / "lat.npy").astype(float)
    lon = np.load(ERA5_ROOT / "lon.npy").astype(float)
    lon = ((lon + 180.0) % 360.0) - 180.0
    conus = ((lat >= 24.0) & (lat <= 50.0))[:, None] & ((lon >= -125.0) & (lon <= -66.0))[None, :]
    globe = np.ones((len(lat), len(lon)), dtype=bool)
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    masks = {"strict_conus": conus, "global": globe, "outside_conus": globe & ~conus}
    return {name: (mask, area) for name, mask in masks.items()}


def sample_path(protocol: str, timestep: int) -> Path:
    matches = sorted(PROTOCOLS[protocol].glob(f"*_t{timestep:04d}_e16_s50.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {protocol} sample file for timestep {timestep}, found {len(matches)}"
        )
    return matches[0]


def members_physical(protocol: str, timestep: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    path = sample_path(protocol, timestep)
    if not path.exists():
        raise FileNotFoundError(path)
    normalized = np.load(path, mmap_mode="r")
    if normalized.shape != (16, 13, 128, 256):
        raise ValueError(f"Unexpected shape {normalized.shape}: {path}")
    return np.asarray(normalized, dtype=np.float32) * std[None] + mean[None]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or np.std(x[good]) == 0 or np.std(y[good]) == 0:
        return float("nan")
    return float(np.corrcoef(x[good], y[good])[0, 1])


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values)
    return float(np.nansum(values * weights) / np.sum(weights[finite]))


def weighted_rmse(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(weighted_mean(np.asarray(values) ** 2, weights)))


def crps_pair(
    samples: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Return cosine-latitude-weighted empirical and fair finite-ensemble CRPS."""
    samples = samples.astype(np.float64, copy=False)
    truth = truth.astype(np.float64, copy=False)
    member_count = samples.shape[0]
    first = np.mean(np.abs(samples - truth[None]), axis=0)
    ordered = np.sort(samples, axis=0)
    coefficient = (2 * np.arange(member_count, dtype=np.float64) - member_count + 1.0)[:, None]
    pair_sum = np.sum(coefficient * ordered, axis=0)
    empirical = first - pair_sum / (member_count * member_count)
    fair = first - pair_sum / (member_count * (member_count - 1))
    return weighted_mean(empirical, weights), weighted_mean(fair, weights)


def evaluate() -> pd.DataFrame:
    mean, std = load_normalization()
    region_supports = supports()
    rows: list[dict[str, object]] = []
    timesteps = load_timesteps()
    for count, timestep in enumerate(timesteps, start=1):
        if count == 1 or count % 25 == 0 or count == len(timesteps):
            print(f"[{count:03d}/{len(timesteps)}] t{timestep:04d}", flush=True)
        truth = load_truth(timestep)
        for protocol in PROTOCOLS:
            ensemble = members_physical(protocol, timestep, mean, std)
            ensemble_mean = ensemble.mean(axis=0)
            ensemble_std = ensemble.std(axis=0, ddof=1)
            q05, q95 = np.quantile(ensemble, [0.05, 0.95], axis=0, method="linear")
            for region, (mask, area) in region_supports.items():
                weights = area[mask]
                for channel, variable in enumerate(VARIABLES):
                    y = truth[channel][mask]
                    mu = ensemble_mean[channel][mask]
                    sigma = ensemble_std[channel][mask]
                    samples = ensemble[:, channel][:, mask]
                    error = mu - y
                    crps, fair_crps = crps_pair(samples, y, weights)
                    rows.append(
                        {
                            "protocol": protocol,
                            "timestep": timestep,
                            "datetime_utc": (
                                pd.Timestamp("2020-01-01T00:00:00Z")
                                + pd.Timedelta(hours=6 * timestep)
                            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "region": region,
                            "variable": variable,
                            "variable_short": SHORT[variable],
                            "is_surface_targeted": variable in SURFACE,
                            "is_aircraft_targeted": variable in AIRCRAFT,
                            "ensemble_mean_rmse": weighted_rmse(error, weights),
                            "spread": weighted_rmse(sigma, weights),
                            "normalized_spread": weighted_rmse(sigma, weights) / std[channel, 0, 0],
                            "coverage_05_95": weighted_mean((y >= q05[channel][mask]) & (y <= q95[channel][mask]), weights),
                            "spread_error_correlation": safe_corr(sigma, np.abs(error)),
                            "crps": crps,
                            "fair_crps": fair_crps,
                        }
                    )
    return pd.DataFrame(rows)


def paired_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = ["timestep", "datetime_utc", "region", "variable", "variable_short", "is_surface_targeted", "is_aircraft_targeted"]
    values = ["ensemble_mean_rmse", "spread", "normalized_spread", "coverage_05_95", "spread_error_correlation", "crps", "fair_crps"]
    wide = metrics.pivot(index=index, columns="protocol", values=values).reset_index()
    wide.columns = ["_".join(x).strip("_") if isinstance(x, tuple) else x for x in wide.columns]
    for metric in ["ensemble_mean_rmse", "spread", "normalized_spread", "crps", "fair_crps"]:
        wide[f"{metric}_pct_change"] = 100.0 * (wide[f"{metric}_R+A+S"] - wide[f"{metric}_R"]) / wide[f"{metric}_R"]
    for metric in ["coverage_05_95", "spread_error_correlation"]:
        wide[f"{metric}_change"] = wide[f"{metric}_R+A+S"] - wide[f"{metric}_R"]

    groups = {
        "all_13_variables": np.ones(len(wide), dtype=bool),
        "surface_targeted_variables": wide["is_surface_targeted"].to_numpy(bool),
        "aircraft_targeted_variables": wide["is_aircraft_targeted"].to_numpy(bool),
    }
    rows = []
    for region in ["strict_conus", "global", "outside_conus"]:
        in_region = wide["region"].eq(region).to_numpy()
        for group, selector in groups.items():
            subset = wide[in_region & selector]
            row: dict[str, object] = {
                "region": region,
                "evaluation_group": group,
                "n_timesteps": int(subset["timestep"].nunique()),
                "n_variables": int(subset["variable"].nunique()),
            }
            for metric in ["ensemble_mean_rmse", "spread", "normalized_spread", "crps", "fair_crps"]:
                row[f"mean_{metric}_pct_change"] = float(subset[f"{metric}_pct_change"].mean())
            for metric in ["coverage_05_95", "spread_error_correlation"]:
                row[f"R_{metric}"] = float(subset[f"{metric}_R"].mean())
                row[f"RAS_{metric}"] = float(subset[f"{metric}_R+A+S"].mean())
                row[f"mean_{metric}_change"] = float(subset[f"{metric}_change"].mean())
            rows.append(row)
    return wide, pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame) -> None:
    strict = summary[summary["region"].eq("strict_conus")].set_index("evaluation_group")
    groups = ["all_13_variables", "surface_targeted_variables", "aircraft_targeted_variables"]
    labels = ["All 13", "Surface-targeted", "Aircraft-targeted"]
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=True)

    x = np.arange(len(groups))
    axes[0].bar(x - 0.18, strict.loc[groups, "mean_ensemble_mean_rmse_pct_change"], 0.36, color="#2B6EA6", label="Ensemble-mean RMSE")
    axes[0].bar(x + 0.18, strict.loc[groups, "mean_crps_pct_change"], 0.36, color="#D97720", label="CRPS")
    axes[0].axhline(0, color="#444444", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("Paired change relative to R (%)")
    axes[0].set_title("Distributional skill")
    axes[0].legend(frameon=True, fontsize=9)

    width = 0.36
    axes[1].bar(x - width / 2, strict.loc[groups, "R_coverage_05_95"], width, color="#8AB6D6", label="R")
    axes[1].bar(x + width / 2, strict.loc[groups, "RAS_coverage_05_95"], width, color="#6AB89A", label="R+A+S")
    axes[1].axhline(0.9, color="#444444", linestyle="--", linewidth=1, label="Large-ensemble 0.90 reference")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Empirical 5%-95% coverage")
    axes[1].set_title("Finite-ensemble coverage")
    axes[1].legend(frameon=True, fontsize=8)

    axes[2].bar(x - width / 2, strict.loc[groups, "R_spread_error_correlation"], width, color="#8AB6D6", label="R")
    axes[2].bar(x + width / 2, strict.loc[groups, "RAS_spread_error_correlation"], width, color="#6AB89A", label="R+A+S")
    axes[2].axhline(0, color="#444444", linestyle="--", linewidth=1)
    axes[2].set_xticks(x, labels, rotation=18, ha="right")
    axes[2].set_ylabel("Spatial correlation")
    axes[2].set_title("Spread-error association")
    axes[2].legend(frameon=True, fontsize=9)

    figure.suptitle("R+A+S selected with 2019 data: full-year 2020 probabilistic diagnostics within CONUS", fontweight="bold")
    figure.savefig(FIGURES / "probabilistic_summary_2020.png", dpi=300)
    figure.savefig(FIGURES / "probabilistic_summary_2020.pdf")
    plt.close(figure)


def write_readme(summary: pd.DataFrame) -> None:
    strict = summary[summary["region"].eq("strict_conus")].set_index("evaluation_group")
    lines = [
        "# Full-Year 2020 Probabilistic Diagnostics",
        "",
        "This report compares the 16-member R-only ensemble with the 16-member R+A+S ensemble selected using 2019 data over all 723 matched 2020 cases.",
        "",
        "## Evidence boundary",
        "",
        "- No GPU sampling is rerun; this is a CPU analysis of the selected-interface arrays.",
        "- All paired changes are formed within timestep and variable before equal-variable averaging.",
        "- Standard empirical CRPS is the manuscript metric. Fair CRPS is included as a fixed-ensemble-size sensitivity check.",
        "- Coverage is diagnostic for the interpolated 5th-95th percentiles of 16 members, not a claim of exact 90% posterior calibration.",
        "",
        "## Strict-CONUS results",
        "",
        "| Evaluation group | Ensemble-mean RMSE change | CRPS change | Fair-CRPS change | R coverage | R+A+S coverage | Coverage change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, label in [
        ("all_13_variables", "All 13 variables"),
        ("surface_targeted_variables", "Surface-targeted variables"),
        ("aircraft_targeted_variables", "Aircraft-targeted variables"),
    ]:
        row = strict.loc[group]
        lines.append(
            f"| {label} | {row['mean_ensemble_mean_rmse_pct_change']:+.3f}% | "
            f"{row['mean_crps_pct_change']:+.3f}% | {row['mean_fair_crps_pct_change']:+.3f}% | "
            f"{row['R_coverage_05_95']:.3f} | {row['RAS_coverage_05_95']:.3f} | "
            f"{100 * row['mean_coverage_05_95_change']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Per-entry metrics: `{TABLES / 'metrics_by_timestep_variable_region_protocol.csv'}`",
            f"- Paired entries: `{TABLES / 'paired_metrics_by_timestep_variable_region.csv'}`",
            f"- Group summary: `{TABLES / 'area_weighted_probabilistic_group_summary.csv'}`",
            f"- Figure: `{FIGURES / 'probabilistic_summary_2020.png'}`",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = evaluate()
    paired, summary = paired_tables(metrics)
    metrics.to_csv(TABLES / "metrics_by_timestep_variable_region_protocol.csv", index=False)
    paired.to_csv(TABLES / "paired_metrics_by_timestep_variable_region.csv", index=False)
    summary.to_csv(TABLES / "area_weighted_probabilistic_group_summary.csv", index=False)
    plot_summary(summary)
    write_readme(summary)
    manifest = {
        "generated_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timestep_manifest": str(TIMESTEP_MANIFEST),
        "n_timesteps": int(metrics["timestep"].nunique()),
        "ensemble_members": 16,
        "steps": 50,
        "protocol_roots": {name: str(root) for name, root in PROTOCOLS.items()},
        "outputs": {"tables": str(TABLES), "figures": str(FIGURES)},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
