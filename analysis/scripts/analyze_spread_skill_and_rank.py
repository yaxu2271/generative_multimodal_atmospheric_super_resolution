#!/usr/bin/env python3
"""Recompute CONUS ensemble-dispersion diagnostics with latitude-area weights.

This is a metric-only audit.  It reads the frozen 723-time R-only and R+A+S
ensembles and does not rerun posterior sampling.
"""

from __future__ import annotations

import json
import h5py
import numpy as np
import pandas as pd

from analysis_paths import output_path, required_path

OUT = output_path("DISPERSION_ANALYSIS_OUTPUT_ROOT", "dispersion_diagnostics")
ERA5 = required_path("ERA5_ROOT")
TIMESTEPS = required_path("EVALUATION_TIMESTEP_MANIFEST")
R_ROOT = required_path("R_ONLY_SAMPLES_ROOT")
RAS_ROOT = required_path("RAS_SAMPLES_ROOT")

VARIABLES = [
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "geopotential_500", "geopotential_850", "u_component_of_wind_500",
    "u_component_of_wind_850", "v_component_of_wind_500",
    "v_component_of_wind_850", "temperature_500", "temperature_850",
    "specific_humidity_500", "specific_humidity_850",
]
SHORT = {
    "2m_temperature": "t2m", "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10", "geopotential_500": "z500",
    "geopotential_850": "z850", "u_component_of_wind_500": "u500",
    "u_component_of_wind_850": "u850", "v_component_of_wind_500": "v500",
    "v_component_of_wind_850": "v850", "temperature_500": "t500",
    "temperature_850": "t850", "specific_humidity_500": "q500",
    "specific_humidity_850": "q850",
}
SURFACE = set(VARIABLES[:3])
AIRCRAFT = {
    "temperature_500", "temperature_850", "u_component_of_wind_500",
    "u_component_of_wind_850", "v_component_of_wind_500", "v_component_of_wind_850",
}
GROUPS = {
    "all_13_variables": VARIABLES,
    "surface_targeted_variables": [v for v in VARIABLES if v in SURFACE],
    "aircraft_targeted_variables": [v for v in VARIABLES if v in AIRCRAFT],
}
PROTOCOLS = {
    "R": (R_ROOT, "igra_only"),
    "R+A+S": (RAS_ROOT, "RplusAplusS_2019_frozen_strict_conus"),
}
N_MEMBERS = 16
N_RANKS = N_MEMBERS + 1
FINITE_ENSEMBLE_FACTOR = np.sqrt((N_MEMBERS + 1.0) / N_MEMBERS)


def load_timesteps() -> list[int]:
    payload = json.loads(TIMESTEPS.read_text())
    values = [int(x) for x in payload.get("timesteps", payload.get("matched_timesteps", []))]
    if len(values) != 723 or len(set(values)) != 723:
        raise RuntimeError(f"Expected 723 unique timesteps, found {len(values)}")
    return values


def normalization() -> tuple[np.ndarray, np.ndarray]:
    means = np.load(ERA5 / "normalize_mean.npz")
    stds = np.load(ERA5 / "normalize_std.npz")
    mean = np.asarray([float(np.asarray(means[v]).reshape(-1)[0]) for v in VARIABLES], dtype=np.float32)
    std = np.asarray([float(np.asarray(stds[v]).reshape(-1)[0]) for v in VARIABLES], dtype=np.float32)
    return mean[:, None, None], std[:, None, None]


def conus_support() -> tuple[np.ndarray, np.ndarray]:
    lat = np.load(ERA5 / "lat.npy").astype(float)
    lon = np.load(ERA5 / "lon.npy").astype(float)
    lon = ((lon + 180.0) % 360.0) - 180.0
    mask = ((lat >= 24.0) & (lat <= 50.0))[:, None] & ((lon >= -125.0) & (lon <= -66.0))[None, :]
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)), dtype=float)
    weights = area[mask]
    return mask, weights / weights.sum()


def sample_path(protocol: str, timestep: int) -> Path:
    root, tag = PROTOCOLS[protocol]
    return root / f"{tag}_t{timestep:04d}_e16_s50.npy"


def tie_safe_ranks(samples: np.ndarray, truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    less = np.sum(samples < truth[None], axis=0)
    equal = np.sum(samples == truth[None], axis=0)
    return less.astype(np.int16) + np.floor(rng.random(truth.size) * (equal + 1)).astype(np.int16)


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    mean, std = normalization()
    mask, weights = conus_support()
    timesteps = load_timesteps()
    names = list(PROTOCOLS)
    rmse_sq_sum = np.zeros((len(names), len(VARIABLES)), dtype=np.float64)
    spread_var_sum = np.zeros_like(rmse_sq_sum)
    coverage_sum = np.zeros_like(rmse_sq_sum)
    rank_counts = np.zeros((len(names), len(VARIABLES), N_RANKS), dtype=np.float64)
    rng = np.random.default_rng(20260801)

    for count, timestep in enumerate(timesteps, start=1):
        if count == 1 or count % 25 == 0 or count == len(timesteps):
            print(f"[{count:03d}/{len(timesteps)}] t{timestep:04d}", flush=True)
        with h5py.File(ERA5 / "test" / f"2020_{timestep:04d}.h5", "r") as handle:
            truth = np.stack([np.asarray(handle["input"][v], dtype=np.float32) for v in VARIABLES])
        for pidx, protocol in enumerate(names):
            normalized = np.load(sample_path(protocol, timestep), mmap_mode="r")
            ensemble = np.asarray(normalized, dtype=np.float32) * std[None] + mean[None]
            ensemble_mean = ensemble.mean(axis=0)
            ensemble_var = ensemble.var(axis=0, ddof=1)
            q05, q95 = np.quantile(ensemble, [0.05, 0.95], axis=0)
            for channel, variable in enumerate(VARIABLES):
                y = truth[channel][mask]
                samples = ensemble[:, channel][:, mask]
                error = ensemble_mean[channel][mask] - y
                rmse_sq_sum[pidx, channel] += float(np.sum(weights * error * error))
                spread_var_sum[pidx, channel] += float(np.sum(weights * ensemble_var[channel][mask]))
                covered = (y >= q05[channel][mask]) & (y <= q95[channel][mask])
                coverage_sum[pidx, channel] += float(np.sum(weights * covered))
                ranks = tie_safe_ranks(samples, y, rng)
                rank_counts[pidx, channel] += np.bincount(ranks, weights=weights, minlength=N_RANKS)

    spread_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for pidx, protocol in enumerate(names):
        for channel, variable in enumerate(VARIABLES):
            rmse = float(np.sqrt(rmse_sq_sum[pidx, channel] / len(timesteps)))
            spread = float(np.sqrt(spread_var_sum[pidx, channel] / len(timesteps)))
            corrected = float(FINITE_ENSEMBLE_FACTOR * spread / rmse)
            spread_rows.append({
                "protocol": protocol, "region": "strict_conus", "variable": variable,
                "variable_short": SHORT[variable], "n_timesteps": len(timesteps),
                "area_weighted_ensemble_mean_rmse": rmse,
                "area_weighted_rms_ensemble_spread": spread,
                "finite_ensemble_corrected_spread_skill_ratio": corrected,
                "area_weighted_coverage_05_95": float(coverage_sum[pidx, channel] / len(timesteps)),
            })
            counts = rank_counts[pidx, channel]
            frequency = counts / counts.sum()
            for rank in range(N_RANKS):
                rank_rows.append({
                    "protocol": protocol, "region": "strict_conus", "variable": variable,
                    "variable_short": SHORT[variable], "rank": rank,
                    "weighted_count": float(counts[rank]), "frequency": float(frequency[rank]),
                    "frequency_relative_to_uniform": float(frequency[rank] * N_RANKS),
                })
    spread = pd.DataFrame(spread_rows)
    ranks = pd.DataFrame(rank_rows)
    spread.to_csv(OUT / "area_weighted_spread_skill_coverage_by_variable.csv", index=False)
    ranks.to_csv(OUT / "area_weighted_rank_histogram_by_variable.csv", index=False)
    return spread, ranks


def summarize(spread: pd.DataFrame, ranks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spread_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        for group, variables in GROUPS.items():
            subset = spread[spread["protocol"].eq(protocol) & spread["variable"].isin(variables)]
            ratios = subset["finite_ensemble_corrected_spread_skill_ratio"].to_numpy(float)
            spread_rows.append({
                "protocol": protocol, "region": "strict_conus", "evaluation_group": group,
                "n_variables": len(variables),
                "mean_corrected_spread_skill_ratio": float(ratios.mean()),
                "min_corrected_spread_skill_ratio": float(ratios.min()),
                "max_corrected_spread_skill_ratio": float(ratios.max()),
                "mean_area_weighted_coverage_05_95": float(subset["area_weighted_coverage_05_95"].mean()),
            })
            grouped = (ranks[ranks["protocol"].eq(protocol) & ranks["variable"].isin(variables)]
                       .groupby("rank", as_index=False)["weighted_count"].sum())
            grouped["frequency"] = grouped["weighted_count"] / grouped["weighted_count"].sum()
            grouped["frequency_relative_to_uniform"] = grouped["frequency"] * N_RANKS
            for row in grouped.itertuples(index=False):
                rank_rows.append({
                    "protocol": protocol, "region": "strict_conus", "evaluation_group": group,
                    "rank": int(row.rank), "weighted_count": float(row.weighted_count),
                    "frequency": float(row.frequency),
                    "frequency_relative_to_uniform": float(row.frequency_relative_to_uniform),
                })
    grouped_spread = pd.DataFrame(spread_rows)
    grouped_ranks = pd.DataFrame(rank_rows)
    grouped_spread.to_csv(OUT / "area_weighted_spread_skill_coverage_by_group.csv", index=False)
    grouped_ranks.to_csv(OUT / "area_weighted_rank_histogram_by_group.csv", index=False)
    return grouped_spread, grouped_ranks


def main() -> None:
    spread, ranks = evaluate()
    grouped_spread, _ = summarize(spread, ranks)
    manifest = {
        "metric": "cosine-latitude area-weighted CONUS ensemble diagnostics",
        "n_timesteps": 723, "n_members": N_MEMBERS,
        "protocols": list(PROTOCOLS),
        "all_spread_skill_ratios_below_one": bool((spread["finite_ensemble_corrected_spread_skill_ratio"] < 1).all()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(grouped_spread.to_string(index=False))


if __name__ == "__main__":
    main()
