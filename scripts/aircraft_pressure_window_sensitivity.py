#!/usr/bin/env python3
"""Pressure-window sensitivity for MADIS aircraft 13-var candidates.

This diagnostic compares altitude-derived pressure bins before aircraft
observations are promoted into a posterior-sampling protocol. It intentionally
does not launch GPU jobs or create sampler inputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DIAG_PATH = SCRIPT_DIR / "diagnose_madis_aircraft_vs_era5.py"


def load_diag_module() -> Any:
    spec = importlib.util.spec_from_file_location("madis_diag", DIAG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {DIAG_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class Window:
    level: int
    name: str
    lo: float
    hi: float


WINDOWS = [
    Window(500, "500_tiny_498_502", 498.0, 502.0),
    Window(500, "500_ultranarrow_495_505", 495.0, 505.0),
    Window(500, "500_narrow_490_510", 490.0, 510.0),
    Window(500, "500_mid_475_525", 475.0, 525.0),
    Window(500, "500_wide_450_550", 450.0, 550.0),
    Window(850, "850_tiny_848_852", 848.0, 852.0),
    Window(850, "850_ultranarrow_845_855", 845.0, 855.0),
    Window(850, "850_narrow_840_860", 840.0, 860.0),
    Window(850, "850_mid_825_875", 825.0, 875.0),
    Window(850, "850_wide_800_900", 800.0, 900.0),
]

REGIONS = {
    "strict_conus": (24.0, 50.0, -125.0, -66.0),
    "north_america": (10.0, 70.0, -170.0, -50.0),
    "global": (-90.0, 90.0, -180.0, 180.0),
}

VARIABLES = [
    ("temperature_500", "temperature", 500),
    ("u_component_of_wind_500", "u", 500),
    ("v_component_of_wind_500", "v", 500),
    ("temperature_850", "temperature", 850),
    ("u_component_of_wind_850", "u", 850),
    ("v_component_of_wind_850", "v", 850),
]

PRODUCT_GROUPS = ["acars", "acarsProfiles", "combined"]


class Accumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sum_obs = 0.0
        self.sum_ref = 0.0
        self.sum_abs_diff = 0.0
        self.sum_diff = 0.0
        self.sum_diff2 = 0.0
        self.sum_obs2 = 0.0
        self.sum_ref2 = 0.0
        self.sum_obs_ref = 0.0
        self.sum_pressure = 0.0
        self.sum_pressure2 = 0.0
        self.cells: set[tuple[int, int]] = set()
        self.source_files: set[str] = set()

    def update(
        self,
        obs: np.ndarray,
        ref: np.ndarray,
        pressure: np.ndarray,
        cells: np.ndarray,
        source_file: str,
    ) -> None:
        mask = np.isfinite(obs) & np.isfinite(ref) & np.isfinite(pressure)
        if not np.any(mask):
            return
        obs = obs[mask].astype(np.float64)
        ref = ref[mask].astype(np.float64)
        pressure = pressure[mask].astype(np.float64)
        cells = cells[mask]
        diff = obs - ref
        self.n += int(obs.size)
        self.sum_obs += float(obs.sum())
        self.sum_ref += float(ref.sum())
        self.sum_abs_diff += float(np.abs(diff).sum())
        self.sum_diff += float(diff.sum())
        self.sum_diff2 += float(np.square(diff).sum())
        self.sum_obs2 += float(np.square(obs).sum())
        self.sum_ref2 += float(np.square(ref).sum())
        self.sum_obs_ref += float((obs * ref).sum())
        self.sum_pressure += float(pressure.sum())
        self.sum_pressure2 += float(np.square(pressure).sum())
        for iy, ix in cells:
            self.cells.add((int(iy), int(ix)))
        self.source_files.add(source_file)

    def row(self, key: tuple[str, str, str, str, int, float, float]) -> dict[str, Any]:
        region, product_group, variable, window_name, level, lo, hi = key
        if self.n == 0:
            return {
                "region": region,
                "product_group": product_group,
                "variable": variable,
                "level_hpa": level,
                "window_name": window_name,
                "pressure_lo_hpa": lo,
                "pressure_hi_hpa": hi,
                "n_obs": 0,
                "covered_era5_cells": 0,
                "mean_points_per_covered_cell": math.nan,
                "bias_obs_minus_era5": math.nan,
                "mae": math.nan,
                "rmse": math.nan,
                "corr": math.nan,
                "obs_mean": math.nan,
                "era5_mean": math.nan,
                "pressure_mean_hpa_alt_derived": math.nan,
                "pressure_std_hpa_alt_derived": math.nan,
                "n_source_files": 0,
            }
        n = float(self.n)
        obs_mean = self.sum_obs / n
        ref_mean = self.sum_ref / n
        cov = self.sum_obs_ref / n - obs_mean * ref_mean
        obs_var = self.sum_obs2 / n - obs_mean * obs_mean
        ref_var = self.sum_ref2 / n - ref_mean * ref_mean
        corr = cov / math.sqrt(obs_var * ref_var) if obs_var > 0 and ref_var > 0 else math.nan
        p_mean = self.sum_pressure / n
        p_var = self.sum_pressure2 / n - p_mean * p_mean
        n_cells = len(self.cells)
        return {
            "region": region,
            "product_group": product_group,
            "variable": variable,
            "level_hpa": level,
            "window_name": window_name,
            "pressure_lo_hpa": lo,
            "pressure_hi_hpa": hi,
            "n_obs": self.n,
            "covered_era5_cells": n_cells,
            "mean_points_per_covered_cell": self.n / n_cells if n_cells else math.nan,
            "bias_obs_minus_era5": self.sum_diff / n,
            "mae": self.sum_abs_diff / n,
            "rmse": math.sqrt(self.sum_diff2 / n),
            "corr": corr,
            "obs_mean": obs_mean,
            "era5_mean": ref_mean,
            "pressure_mean_hpa_alt_derived": p_mean,
            "pressure_std_hpa_alt_derived": math.sqrt(max(0.0, p_var)),
            "n_source_files": len(self.source_files),
        }


def lon_to_180(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def cell_indices(lat_grid: np.ndarray, lon_grid: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lats = np.asarray(lat_grid, dtype=np.float64)
    lons = np.asarray(lon_grid, dtype=np.float64)
    dlat = abs(float(np.median(np.diff(lats))))
    dlon = abs(float(np.median(np.diff(lons))))
    if lats[0] > lats[-1]:
        iy = np.rint((lats[0] - lat) / dlat).astype(int)
    else:
        iy = np.rint((lat - lats[0]) / dlat).astype(int)
    lon360 = np.mod(lon, 360.0)
    ix = np.rint(lon360 / dlon).astype(int) % len(lons)
    iy = np.clip(iy, 0, len(lats) - 1)
    return np.stack([iy, ix], axis=1)


def process_file(path: Path, diag: Any, era5_root: Path, lat_grid: np.ndarray, lon_grid: np.ndarray, acc: dict[tuple, Accumulator]) -> None:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise SystemExit("netCDF4 is required") from exc

    product = "acarsProfiles" if "acarsProfiles" in str(path) else "acars"
    dt = diag.parse_datetime(path)
    t_index = diag.era5_timestep_index(dt)
    tmp = diag.decode_to_temp(path)
    try:
        with Dataset(tmp, "r") as ds:
            temp = diag.read_var(ds, "temperature")
            wind_speed = diag.read_var(ds, "windSpeed")
            wind_dir = diag.read_var(ds, "windDir")
            altitude = diag.read_var(ds, "altitude")
            if temp is None or altitude is None:
                return
            lat2, lon2, _location_source = diag.read_locations(ds, temp.shape)
            if lat2 is None or lon2 is None:
                return
            lon2 = lon_to_180(lon2)
            pressure = diag.pressure_hpa_from_altitude_m(altitude)
            u, v = diag.to_uv(wind_speed, wind_dir)
            loc_qc = (
                (diag.read_qcr(ds, "latitude", temp.shape) == 0)
                & (diag.read_qcr(ds, "longitude", temp.shape) == 0)
                & (diag.read_qcr(ds, "altitude", temp.shape) == 0)
            )
            temp_qc = (diag.read_qcr(ds, "temperature", temp.shape) == 0) & (temp > 180.0) & (temp < 330.0)
            wind_qc = (
                (diag.read_qcr(ds, "windSpeed", temp.shape) == 0)
                & (diag.read_qcr(ds, "windDir", temp.shape) == 0)
                & np.isfinite(wind_speed)
                & np.isfinite(wind_dir)
                & (wind_speed >= 0.0)
                & (wind_speed < 150.0)
            )
            base = (
                loc_qc
                & np.isfinite(lat2)
                & np.isfinite(lon2)
                & np.isfinite(pressure)
                & (pressure > 100.0)
                & (pressure < 1050.0)
            )
            obs_by_kind = {
                "temperature": temp,
                "u": u if u is not None else np.full_like(temp, np.nan),
                "v": v if v is not None else np.full_like(temp, np.nan),
            }
            qc_by_kind = {
                "temperature": temp_qc,
                "u": wind_qc,
                "v": wind_qc,
            }
            era5_cache: dict[str, np.ndarray] = {}
            for variable, kind, level in VARIABLES:
                obs = obs_by_kind[kind]
                if variable not in era5_cache:
                    era5_cache[variable] = diag.load_era5_field(era5_root, t_index, variable)
                for window in WINDOWS:
                    if window.level != level:
                        continue
                    pwin = (pressure >= window.lo) & (pressure <= window.hi)
                    for region, box in REGIONS.items():
                        region_mask = diag.in_box(lat2, lon2, box)
                        mask = base & qc_by_kind[kind] & pwin & region_mask & np.isfinite(obs)
                        if not np.any(mask):
                            continue
                        ref = diag.interp_regular_grid(era5_cache[variable], lat_grid, lon_grid, lat2[mask], lon2[mask])
                        cells = cell_indices(lat_grid, lon_grid, lat2[mask], lon2[mask])
                        for product_group in [product, "combined"]:
                            key = (region, product_group, variable, window.name, level, window.lo, window.hi)
                            acc[key].update(obs[mask], ref, pressure[mask], cells, str(path))
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def plot_summary(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = df[(df["product_group"] == "combined") & (df["region"] == "strict_conus")].copy()
    if primary.empty:
        return
    order = [w.name for w in WINDOWS]
    for metric, ylabel in [("rmse", "RMSE"), ("n_obs", "QCR==0 obs count"), ("covered_era5_cells", "covered ERA5 cells")]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.8), sharex=False)
        axes = axes.ravel()
        for ax, variable in zip(axes, [v[0] for v in VARIABLES]):
            sub = primary[primary["variable"] == variable].set_index("window_name").reindex(order).dropna(subset=[metric])
            if sub.empty:
                ax.axis("off")
                continue
            ax.plot(sub.index, sub[metric], marker="o", linewidth=2.2)
            ax.set_title(variable, fontsize=10, weight="bold")
            ax.tick_params(axis="x", labelrotation=35, labelsize=8)
            ax.grid(alpha=0.25)
            ax.set_ylabel(ylabel)
        fig.suptitle(f"MADIS aircraft pressure-window sensitivity: {ylabel} (strict CONUS, combined)", weight="bold")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(out_dir / f"madis_aircraft_pressure_window_sensitivity_strict_conus_combined_{metric}.png", dpi=230)
        plt.close(fig)

    # Product comparison for the mid/default windows.
    mid = df[
        (df["region"] == "strict_conus")
        & (df["window_name"].isin(["500_mid_475_525", "850_mid_825_875"]))
    ].copy()
    fig, ax = plt.subplots(figsize=(12, 6.2))
    labels = []
    x = []
    y = []
    colors = []
    color_map = {"acars": "#1f77b4", "acarsProfiles": "#9467bd", "combined": "#2ca02c"}
    i = 0
    for variable in [v[0] for v in VARIABLES]:
        for product in PRODUCT_GROUPS:
            sub = mid[(mid["variable"] == variable) & (mid["product_group"] == product)]
            if sub.empty:
                continue
            x.append(i)
            y.append(float(sub["rmse"].iloc[0]))
            colors.append(color_map[product])
            labels.append(variable.replace("_component_of_wind", "").replace("_", "\n"))
            i += 1
        i += 0.8
    ax.bar(x, y, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel("RMSE")
    ax.set_title("MADIS aircraft product comparison at default pressure windows (strict CONUS)", weight="bold")
    handles = [plt.Line2D([0], [0], color=color_map[p], lw=8, label=p) for p in PRODUCT_GROUPS]
    ax.legend(handles=handles)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "madis_aircraft_pressure_window_product_comparison_default_windows.png", dpi=230)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--era5-root", type=Path, default=Path("/depot/rmaulik/data/yangxu/era5_subset"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    diag = load_diag_module()
    lat_grid = np.load(args.era5_root / "lat.npy").astype(np.float64)
    lon_grid = np.load(args.era5_root / "lon.npy").astype(np.float64)
    acc: dict[tuple, Accumulator] = defaultdict(Accumulator)
    for path in args.files:
        process_file(path, diag, args.era5_root, lat_grid, lon_grid, acc)

    rows = [value.row(key) for key, value in sorted(acc.items())]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / "madis_aircraft_pressure_window_sensitivity_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    df = pd.DataFrame(rows)
    plot_summary(df, args.out_dir)
    print(f"Wrote {summary_csv}")
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
