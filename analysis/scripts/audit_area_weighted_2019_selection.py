#!/usr/bin/env python3
"""Replay the 2019 Stage-IV joint selection with area-weighted CONUS RMSE."""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import h5py
import numpy as np
import pandas as pd

from analysis_paths import output_path, required_path

OUT = output_path("AREA_WEIGHTED_SELECTION_OUTPUT_ROOT", "area_weighted_2019_selection")
ERA5 = required_path("ERA5_ROOT")
STAGE4_RUN = required_path("STAGE4_CANDIDATE_RUN_ROOT")
BASELINE_ROOT = required_path("STAGE4_R_ONLY_SAMPLES_ROOT")
OLD_RANKING = required_path("EQUAL_GRIDPOINT_2019_RANKING_CSV")
VARIABLES = [
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "geopotential_500", "geopotential_850", "u_component_of_wind_500",
    "u_component_of_wind_850", "v_component_of_wind_500",
    "v_component_of_wind_850", "temperature_500", "temperature_850",
    "specific_humidity_500", "specific_humidity_850",
]
SURFACE = set(VARIABLES[:3])
AIRCRAFT = {
    "temperature_500", "temperature_850", "u_component_of_wind_500",
    "u_component_of_wind_850", "v_component_of_wind_500", "v_component_of_wind_850",
}
SAMPLE_RE = re.compile(r"_t(\d{4})_e16_s50\.npy$")
REPLICATES = 10_000
SEED = 1701


@lru_cache(maxsize=1)
def normalization() -> tuple[np.ndarray, np.ndarray]:
    means = np.load(ERA5 / "normalize_mean.npz")
    stds = np.load(ERA5 / "normalize_std.npz")
    mean = np.asarray([float(np.asarray(means[v]).reshape(-1)[0]) for v in VARIABLES], dtype=np.float32)
    std = np.asarray([float(np.asarray(stds[v]).reshape(-1)[0]) for v in VARIABLES], dtype=np.float32)
    return mean[:, None, None], std[:, None, None]


@lru_cache(maxsize=1)
def conus_support() -> tuple[np.ndarray, np.ndarray]:
    lat = np.load(ERA5 / "lat.npy").astype(float)
    lon = np.load(ERA5 / "lon.npy").astype(float)
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    mask = ((lat >= 24.0) & (lat <= 50.0))[:, None] & ((lon180 >= -125.0) & (lon180 <= -66.0))[None, :]
    weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)), dtype=float)
    return mask, weights


def sample_map(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in root.glob("samples/*/*.npy"):
        match = SAMPLE_RE.search(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def protocols() -> pd.DataFrame:
    rows = []
    for directory in sorted((STAGE4_RUN / "protocols").iterdir()):
        if not directory.is_dir():
            continue
        resolved = json.loads((directory / "resolved_protocol.json").read_text())
        protocol = resolved["protocol"]
        paths = sample_map(directory)
        rows.append({
            "protocol_id": protocol["protocol_id"],
            "aircraft_candidate": protocol["aircraft_candidate"]["name"],
            "surface_candidate": protocol["surface_candidate"]["name"],
            "aircraft_lambda": float(protocol["aircraft_parameters"]["lambda"]),
            "surface_lambda": float(protocol["surface_parameters"]["lambda"]),
            "paths": paths,
        })
    frame = pd.DataFrame(rows)
    if len(frame) != 9:
        raise RuntimeError(f"Expected 9 joint candidates, found {len(frame)}")
    return frame


def evaluate_pair(payload: tuple[str, int, str, str]) -> list[dict[str, object]]:
    protocol_id, timestep, baseline_path, candidate_path = payload
    mean, std = normalization()
    mask, weights = conus_support()
    weight = weights[mask]
    baseline = np.asarray(np.load(baseline_path, mmap_mode="r"), dtype=np.float32).mean(axis=0) * std + mean
    candidate = np.asarray(np.load(candidate_path, mmap_mode="r"), dtype=np.float32).mean(axis=0) * std + mean
    with h5py.File(ERA5 / "val" / f"2019_{timestep:04d}.h5", "r") as handle:
        truth = np.stack([np.asarray(handle["input"][v], dtype=np.float32) for v in VARIABLES])
    baseline_sq = (baseline[:, mask] - truth[:, mask]) ** 2
    candidate_sq = (candidate[:, mask] - truth[:, mask]) ** 2
    baseline_rmse = np.sqrt(np.sum(baseline_sq * weight[None], axis=1) / np.sum(weight))
    candidate_rmse = np.sqrt(np.sum(candidate_sq * weight[None], axis=1) / np.sum(weight))
    timestamp = pd.Timestamp("2019-01-01T00:00:00Z") + pd.Timedelta(hours=6 * timestep)
    return [{
        "protocol_id": protocol_id,
        "timestep": timestep,
        "datetime_utc": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "month": int(timestamp.month),
        "variable": variable,
        "baseline_rmse": float(baseline_rmse[index]),
        "candidate_rmse": float(candidate_rmse[index]),
        "pct_change": float(100.0 * (candidate_rmse[index] - baseline_rmse[index]) / baseline_rmse[index]),
    } for index, variable in enumerate(VARIABLES)]


def paired_month_bootstrap(by_time: pd.DataFrame, raw_best: str) -> pd.DataFrame:
    # Each calendar month contributes its paired 1st/15th development cases.
    pivot = by_time.pivot_table(
        index="month", columns="protocol_id", values="all13", aggfunc="mean"
    )
    if pivot.shape[0] != 12 or pivot.isna().any().any():
        raise RuntimeError(f"Expected complete 12-month paired table, found {pivot.shape}")
    rng = np.random.default_rng(SEED)
    month_indices = rng.integers(0, 12, size=(REPLICATES, 12))
    rows = []
    for candidate in pivot.columns:
        delta = pivot[candidate].to_numpy(float) - pivot[raw_best].to_numpy(float)
        boot = delta[month_indices].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append({
            "protocol_id": candidate,
            "raw_best": raw_best,
            "observed_mean_difference_pp": float(delta.mean()),
            "ci95_low_pp": float(low),
            "ci95_high_pp": float(high),
            "tied_with_raw_best": bool(low <= 0.0 <= high),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_frame = protocols()
    baseline_paths = sample_map(BASELINE_ROOT)
    payloads = []
    for row in protocol_frame.itertuples(index=False):
        if set(row.paths) != set(baseline_paths):
            raise RuntimeError(f"Timestep mismatch for {row.protocol_id}")
        payloads.extend((row.protocol_id, t, str(baseline_paths[t]), str(row.paths[t])) for t in sorted(row.paths))
    rows = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for index, result in enumerate(executor.map(evaluate_pair, payloads, chunksize=1), start=1):
            rows.extend(result)
            if index == 1 or index % 24 == 0 or index == len(payloads):
                print(f"[{index}/{len(payloads)}]", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "area_weighted_2019_metrics.csv", index=False)
    grouped_rows = []
    for keys, frame in metrics.groupby(["protocol_id", "timestep", "datetime_utc", "month"]):
        grouped_rows.append({
            "protocol_id": keys[0], "timestep": keys[1], "datetime_utc": keys[2], "month": keys[3],
            "all13": float(frame["pct_change"].mean()),
            "surface3": float(frame.loc[frame["variable"].isin(SURFACE), "pct_change"].mean()),
            "aircraft6": float(frame.loc[frame["variable"].isin(AIRCRAFT), "pct_change"].mean()),
        })
    by_time = pd.DataFrame(grouped_rows)
    by_time.to_csv(OUT / "area_weighted_2019_effects_by_time.csv", index=False)
    ranking = by_time.groupby("protocol_id", as_index=False).agg(
        all13_mean=("all13", "mean"), surface3_mean=("surface3", "mean"), aircraft6_mean=("aircraft6", "mean")
    ).merge(protocol_frame.drop(columns="paths"), on="protocol_id", validate="one_to_one")
    ranking = ranking.sort_values("all13_mean").reset_index(drop=True)
    ranking["area_weighted_rank"] = np.arange(1, len(ranking) + 1)
    old = pd.read_csv(OLD_RANKING)[["protocol_id", "all_13_variables_mean", "all13_rank"]].rename(
        columns={"all_13_variables_mean": "equal_gridpoint_all13_mean", "all13_rank": "equal_gridpoint_rank"}
    )
    ranking = ranking.merge(old, on="protocol_id", validate="one_to_one")
    ranking.to_csv(OUT / "area_weighted_2019_candidate_ranking.csv", index=False)
    raw_best = str(ranking.iloc[0]["protocol_id"])
    bootstrap = paired_month_bootstrap(by_time, raw_best)
    bootstrap.to_csv(OUT / "area_weighted_paired_month_bootstrap.csv", index=False)
    tied = set(bootstrap.loc[bootstrap["tied_with_raw_best"], "protocol_id"])
    selected = "joint3x3__a_2019_best__s_compromise" if "joint3x3__a_2019_best__s_compromise" in tied else None
    summary = {
        "metric": "cosine-latitude area-weighted strict-CONUS RMSE",
        "candidate_count": 9,
        "development_timesteps": 24,
        "raw_best_protocol": raw_best,
        "tied_protocols": sorted(tied),
        "prespecified_center_protocol": "joint3x3__a_2019_best__s_compromise",
        "selected_protocol_under_existing_tie_rule": selected,
        "original_selected_protocol": "joint3x3__a_2019_best__s_compromise",
        "selection_unchanged": selected == "joint3x3__a_2019_best__s_compromise",
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": SEED,
    }
    (OUT / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(ranking[["protocol_id", "all13_mean", "area_weighted_rank", "equal_gridpoint_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
