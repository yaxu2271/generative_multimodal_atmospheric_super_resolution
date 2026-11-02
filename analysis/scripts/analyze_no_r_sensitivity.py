#!/usr/bin/env python3
"""Area-weighted ERA5 evaluation for the frozen-policy 24-time no-R pilot."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


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
PROTOCOL_TAGS = {
    "prior_no_observations": "prior_no_observations",
    "aircraft_only": "aircraft_only",
    "surface_only": "surface_only",
    "aircraft_surface": "aircraft_surface",
}
SAMPLE_RE = re.compile(r"_t(\d{4})_e16_s50\.npy$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed-parameter 24-case no-R sensitivity analysis."
    )
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--r-only-samples-root", type=Path, required=True)
    parser.add_argument("--era5-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sample_maps(protocol_root: Path, r_root: Path) -> dict[str, dict[int, Path]]:
    result: dict[str, dict[int, Path]] = {}
    for protocol, tag in PROTOCOL_TAGS.items():
        mapping = {}
        for path in (protocol_root / protocol / "samples" / tag).glob("*_e16_s50.npy"):
            match = SAMPLE_RE.search(path.name)
            if match:
                mapping[int(match.group(1))] = path
        result[protocol] = mapping
    timesteps = set(next(iter(result.values())))
    if len(timesteps) != 24 or any(set(mapping) != timesteps for mapping in result.values()):
        raise RuntimeError("Expected the same 24 timesteps for all no-R protocols")
    result["radiosonde_only"] = {
        timestep: r_root / f"igra_only_t{timestep:04d}_e16_s50.npy" for timestep in timesteps
    }
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    means = np.load(args.era5_root / "normalize_mean.npz")
    stds = np.load(args.era5_root / "normalize_std.npz")
    mean = np.asarray([float(np.asarray(means[v]).reshape(-1)[0]) for v in VARIABLES], dtype=np.float32)[:, None, None]
    std = np.asarray([float(np.asarray(stds[v]).reshape(-1)[0]) for v in VARIABLES], dtype=np.float32)[:, None, None]
    lat = np.load(args.era5_root / "lat.npy").astype(float)
    lon = np.load(args.era5_root / "lon.npy").astype(float)
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    mask = ((lat >= 24.0) & (lat <= 50.0))[:, None] & ((lon180 >= -125.0) & (lon180 <= -66.0))[None, :]
    area = (np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon))))[mask]
    denominator = float(area.sum())
    maps = sample_maps(args.protocol_root, args.r_only_samples_root)
    rows = []
    for protocol, mapping in maps.items():
        for timestep, path in sorted(mapping.items()):
            if not path.exists():
                raise FileNotFoundError(path)
            prediction = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32).mean(axis=0) * std + mean
            with h5py.File(args.era5_root / "test" / f"2020_{timestep:04d}.h5", "r") as handle:
                truth = np.stack([np.asarray(handle["input"][v], dtype=np.float32) for v in VARIABLES])
            squared = (prediction[:, mask] - truth[:, mask]) ** 2
            rmse = np.sqrt(np.sum(squared * area[None], axis=1) / denominator)
            for index, variable in enumerate(VARIABLES):
                rows.append({"protocol": protocol, "timestep": timestep, "variable": variable, "rmse": float(rmse[index])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "no_r_area_weighted_metrics.csv", index=False)
    baseline = metrics[metrics["protocol"].eq("radiosonde_only")][["timestep", "variable", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
    paired = metrics.merge(baseline, on=["timestep", "variable"], validate="many_to_one")
    paired["pct_change_vs_radiosonde_only"] = 100.0 * (paired["rmse"] - paired["baseline_rmse"]) / paired["baseline_rmse"]
    paired.to_csv(args.output_dir / "no_r_area_weighted_paired.csv", index=False)
    summaries = []
    for protocol, frame in paired.groupby("protocol"):
        summaries.append({
            "protocol": protocol,
            "n_timesteps": int(frame["timestep"].nunique()),
            "all13_pct_change_vs_R": float(frame["pct_change_vs_radiosonde_only"].mean()),
            "surface3_pct_change_vs_R": float(frame.loc[frame["variable"].isin(SURFACE), "pct_change_vs_radiosonde_only"].mean()),
            "aircraft6_pct_change_vs_R": float(frame.loc[frame["variable"].isin(AIRCRAFT), "pct_change_vs_radiosonde_only"].mean()),
        })
    summary = pd.DataFrame(summaries).sort_values("all13_pct_change_vs_R")
    summary.to_csv(args.output_dir / "no_r_area_weighted_summary.csv", index=False)
    payload = {
        "metric": "cosine-latitude area-weighted strict-CONUS ensemble-mean RMSE",
        "n_timesteps": 24,
        "interpretive_boundary": "Frozen A/S parameters selected with R present; this is a removal ablation, not an optimized no-R or kriging baseline.",
    }
    (args.output_dir / "no_r_analysis_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
