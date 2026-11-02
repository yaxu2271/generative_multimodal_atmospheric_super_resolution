#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_paths import output_path, required_path

ERA5_ROOT = required_path("ERA5_ROOT")
BASELINE_ROOT = required_path("R_ONLY_SAMPLES_ROOT")
RAS_SAMPLES_ROOT = required_path("RAS_SAMPLES_ROOT")
TIMESTEP_MANIFEST = required_path("EVALUATION_TIMESTEP_MANIFEST")
OUT_DIR = output_path("ANNUAL_RMSE_OUTPUT_ROOT", "annual_rmse")
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

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

UNITS = {
    "2m_temperature": "K",
    "10m_u_component_of_wind": "m/s",
    "10m_v_component_of_wind": "m/s",
    "geopotential_500": "m^2/s^2",
    "geopotential_850": "m^2/s^2",
    "u_component_of_wind_500": "m/s",
    "u_component_of_wind_850": "m/s",
    "v_component_of_wind_500": "m/s",
    "v_component_of_wind_850": "m/s",
    "temperature_500": "K",
    "temperature_850": "K",
    "specific_humidity_500": "kg/kg",
    "specific_humidity_850": "kg/kg",
}

SURFACE3 = {
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
}

AIRCRAFT6 = {
    "temperature_500",
    "temperature_850",
    "u_component_of_wind_500",
    "u_component_of_wind_850",
    "v_component_of_wind_500",
    "v_component_of_wind_850",
}

TIMESTEP_RE = re.compile(r"_t(\d{4})_e16_s50\.npy$")


@dataclass(frozen=True)
class Protocol:
    label: str
    sample_dir: Path
    display: str
    group: str
    lambda_aircraft: float | None
    std_aircraft: float | None
    gamma_aircraft: float | None
    lambda_surface: float | None
    std_surface: float | None
    gamma_surface: float | None


PROTOCOLS = [
    Protocol(
        label="RplusAplusS_selected_2019",
        sample_dir=RAS_SAMPLES_ROOT,
        display="R+A+S (selected with 2019 data)",
        group="R+A+S",
        lambda_aircraft=0.4,
        std_aircraft=5e-4,
        gamma_aircraft=2e-5,
        lambda_surface=0.4,
        std_surface=1.25e-4,
        gamma_surface=4e-5,
    ),
]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_timesteps() -> list[int]:
    payload = json.loads(TIMESTEP_MANIFEST.read_text())
    timesteps = [int(t) for t in payload["timesteps"]]
    if len(timesteps) != 723:
        raise RuntimeError(f"Expected 723 matched timesteps, got {len(timesteps)}")
    return timesteps


def load_grid_supports() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    lats = np.load(ERA5_ROOT / "lat.npy").astype(float)
    lons = np.load(ERA5_ROOT / "lon.npy").astype(float)
    lon180 = ((lons + 180.0) % 360.0) - 180.0
    strict = (
        ((lats >= 24.0) & (lats <= 50.0))[:, None]
        & ((lon180 >= -125.0) & (lon180 <= -66.0))[None, :]
    )
    global_mask = np.ones((len(lats), len(lons)), dtype=bool)
    area_weights = np.cos(np.deg2rad(lats))[:, None] * np.ones((1, len(lons)))
    masks = {
        "strict_conus": strict,
        "global": global_mask,
        "global_excluding_strict_conus": global_mask & ~strict,
    }
    return {name: (mask, area_weights) for name, mask in masks.items()}


def load_norm() -> tuple[np.ndarray, np.ndarray]:
    means = np.load(ERA5_ROOT / "normalize_mean.npz")
    stds = np.load(ERA5_ROOT / "normalize_std.npz")
    mean = np.asarray([float(np.asarray(means[v]).reshape(-1)[0]) for v in VARIABLES], dtype=np.float32)
    std = np.asarray([float(np.asarray(stds[v]).reshape(-1)[0]) for v in VARIABLES], dtype=np.float32)
    return mean[:, None, None], std[:, None, None]


def load_truth(timestep: int) -> np.ndarray:
    fields = []
    with h5py.File(ERA5_ROOT / "test" / f"2020_{timestep:04d}.h5", "r") as f:
        for var in VARIABLES:
            fields.append(np.asarray(f["input"][var], dtype=np.float32))
    return np.stack(fields, axis=0)


def load_sample_mean(path: Path, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    return (np.asarray(arr, dtype=np.float32).mean(axis=0) * std + mean).astype(np.float32)


def rmse(
    field: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    area_weights: np.ndarray,
) -> np.ndarray:
    diff = field[:, mask] - truth[:, mask]
    squared = np.asarray(diff * diff, dtype=np.float64)
    weights = area_weights[mask]
    finite = np.isfinite(squared)
    numerator = np.nansum(squared * weights[None, :], axis=1)
    denominator = np.sum(finite * weights[None, :], axis=1)
    return np.sqrt(numerator / denominator)


def datetime_utc(timestep: int) -> pd.Timestamp:
    return pd.Timestamp("2020-01-01T00:00:00Z") + pd.Timedelta(hours=6 * timestep)


def sample_paths(protocol: Protocol) -> dict[int, Path]:
    paths = {}
    for p in sorted(protocol.sample_dir.glob("*.npy")):
        match = TIMESTEP_RE.search(p.name)
        if match:
            paths[int(match.group(1))] = p
    return paths


def sample_path(root: Path, timestep: int) -> Path:
    matches = sorted(root.glob(f"*_t{timestep:04d}_e16_s50.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one sample file for timestep {timestep} in {root}, found {len(matches)}"
        )
    return matches[0]


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    timesteps = load_timesteps()
    supports = load_grid_supports()
    mean, std = load_norm()

    protocol_paths = {p.label: sample_paths(p) for p in PROTOCOLS}
    sample_counts = {p.label: len(protocol_paths[p.label]) for p in PROTOCOLS}
    for protocol in PROTOCOLS:
        missing = sorted(set(timesteps) - set(protocol_paths[protocol.label]))
        if missing:
            raise FileNotFoundError(f"{protocol.label} missing {len(missing)} timesteps: {missing[:20]}")

    rows = []
    for idx, timestep in enumerate(timesteps, start=1):
        if idx == 1 or idx % 50 == 0 or idx == len(timesteps):
            print(f"[{idx:03d}/{len(timesteps)}] t{timestep:04d}", flush=True)
        truth = load_truth(timestep)
        baseline_path = sample_path(BASELINE_ROOT, timestep)
        baseline = load_sample_mean(baseline_path, mean, std)
        baseline_rmse = {
            region: rmse(baseline, truth, mask, weights)
            for region, (mask, weights) in supports.items()
        }
        dt = datetime_utc(timestep)
        for protocol in PROTOCOLS:
            pred = load_sample_mean(protocol_paths[protocol.label][timestep], mean, std)
            for region, (mask, weights) in supports.items():
                method = rmse(pred, truth, mask, weights)
                base = baseline_rmse[region]
                pct_delta = 100.0 * (method - base) / base
                for i, var in enumerate(VARIABLES):
                    rows.append({
                        "protocol": protocol.label,
                        "display": protocol.display,
                        "group": protocol.group,
                        "lambda_aircraft": protocol.lambda_aircraft,
                        "std_aircraft": protocol.std_aircraft,
                        "gamma_aircraft": protocol.gamma_aircraft,
                        "lambda_surface": protocol.lambda_surface,
                        "std_surface": protocol.std_surface,
                        "gamma_surface": protocol.gamma_surface,
                        "sample_directory": str(protocol.sample_dir),
                        "timestep": timestep,
                        "datetime_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "month": int(dt.month),
                        "region": region,
                        "variable": var,
                        "var_short": SHORT[var],
                        "unit": UNITS[var],
                        "is_surface3": var in SURFACE3,
                        "is_aircraft6": var in AIRCRAFT6,
                        "baseline_rmse": float(base[i]),
                        "method_rmse": float(method[i]),
                        "rmse_change_pct_vs_R": float(pct_delta[i]),
                    })

    df = pd.DataFrame(rows)
    keys = [
        "region", "protocol", "display", "group",
        "lambda_aircraft", "std_aircraft", "gamma_aircraft",
        "lambda_surface", "std_surface", "gamma_surface", "sample_directory",
    ]
    summary = (
        df.groupby(keys, as_index=False, dropna=False)
        .agg(
            n_eval_timesteps=("timestep", lambda x: int(pd.Series(x).nunique())),
            all13_mean_rmse_change_pct=("rmse_change_pct_vs_R", "mean"),
            all13_median_rmse_change_pct=("rmse_change_pct_vs_R", "median"),
            all13_improved_frac=("rmse_change_pct_vs_R", lambda x: float(np.mean(np.asarray(x) < 0.0))),
        )
    )
    surface = (
        df[df["is_surface3"]]
        .groupby(keys, as_index=False, dropna=False)
        .agg(
            surface3_mean_rmse_change_pct=("rmse_change_pct_vs_R", "mean"),
            surface3_median_rmse_change_pct=("rmse_change_pct_vs_R", "median"),
            surface3_improved_frac=("rmse_change_pct_vs_R", lambda x: float(np.mean(np.asarray(x) < 0.0))),
        )
    )
    constrained = (
        df[df["is_aircraft6"]]
        .groupby(keys, as_index=False, dropna=False)
        .agg(
            aircraft6_mean_rmse_change_pct=("rmse_change_pct_vs_R", "mean"),
            aircraft6_median_rmse_change_pct=("rmse_change_pct_vs_R", "median"),
            aircraft6_improved_frac=("rmse_change_pct_vs_R", lambda x: float(np.mean(np.asarray(x) < 0.0))),
        )
    )
    summary = summary.merge(surface, how="left").merge(constrained, how="left")
    meta = {
        "ras_samples_root": str(RAS_SAMPLES_ROOT),
        "baseline_root": str(BASELINE_ROOT),
        "era5_root": str(ERA5_ROOT),
        "timestep_manifest": str(TIMESTEP_MANIFEST),
        "n_timesteps": len(timesteps),
        "first_timestep": min(timesteps),
        "last_timestep": max(timesteps),
        "sample_counts": sample_counts,
    }
    return df, summary, meta


def aggregate_set(df: pd.DataFrame, var_filter: str) -> pd.DataFrame:
    if var_filter == "all13":
        sub = df
    elif var_filter == "surface3":
        sub = df[df["is_surface3"]]
    elif var_filter == "aircraft6":
        sub = df[df["is_aircraft6"]]
    else:
        raise ValueError(var_filter)
    return (
        sub.groupby(["region", "protocol", "display", "timestep", "datetime_utc", "month"], as_index=False)
        .agg(
            rmse_change_pct=("rmse_change_pct_vs_R", "mean"),
            baseline_rmse=("baseline_rmse", "mean"),
            method_rmse=("method_rmse", "mean"),
        )
    )


def write_result_tables(df: pd.DataFrame) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    per_var_rows = []
    for (region, protocol, display, variable), g in df.groupby(["region", "protocol", "display", "variable"]):
        per_var_rows.append({
            "region": region,
            "protocol": protocol,
            "display": display,
            "variable": variable,
            "var_short": SHORT[variable],
            "unit": UNITS[variable],
            "baseline_rmse_mean": g["baseline_rmse"].mean(),
            "baseline_rmse_sd": g["baseline_rmse"].std(ddof=1),
            "method_rmse_mean": g["method_rmse"].mean(),
            "method_rmse_sd": g["method_rmse"].std(ddof=1),
            "mean_rmse_change_pct": g["rmse_change_pct_vs_R"].mean(),
            "median_rmse_change_pct": g["rmse_change_pct_vs_R"].median(),
            "improved_timestep_frac": float(np.mean(g["rmse_change_pct_vs_R"].to_numpy() < 0.0)),
        })
    per_var = pd.DataFrame(per_var_rows)
    out = TABLE_DIR / "full_year_per_variable_summary.csv"
    per_var.to_csv(out, index=False)
    outputs["per_variable_summary"] = out

    final = per_var[per_var["protocol"].eq("RplusAplusS_selected_2019")].copy()
    for region in ["strict_conus", "global", "global_excluding_strict_conus"]:
        region_df = final[final["region"].eq(region)].copy()
        region_df["sort_key"] = region_df["mean_rmse_change_pct"]
        region_df = region_df.sort_values("sort_key")
        out = TABLE_DIR / f"selected_RAS_result_table_{region}.csv"
        region_df.drop(columns=["sort_key"]).to_csv(out, index=False)
        outputs[f"selected_RAS_result_table_{region}"] = out

    monthly_rows = []
    for var_set in ["all13", "surface3", "aircraft6"]:
        agg = aggregate_set(df, var_set)
        monthly = (
            agg.groupby(["region", "protocol", "display", "month"], as_index=False)
            .agg(
                mean_rmse_change_pct=("rmse_change_pct", "mean"),
                median_rmse_change_pct=("rmse_change_pct", "median"),
                improved_timestep_frac=("rmse_change_pct", lambda x: float(np.mean(np.asarray(x) < 0.0))),
                n_timesteps=("timestep", "nunique"),
            )
        )
        monthly["var_set"] = var_set
        monthly_rows.append(monthly)
    monthly_all = pd.concat(monthly_rows, ignore_index=True)
    out = TABLE_DIR / "full_year_monthly_summary.csv"
    monthly_all.to_csv(out, index=False)
    outputs["monthly_summary"] = out

    return outputs


def plot_metric_matrix(summary: pd.DataFrame, region: str, out: Path) -> None:
    metrics = [
        ("all13_mean_rmse_change_pct", "All 13 variables"),
        ("surface3_mean_rmse_change_pct", "Surface-targeted"),
        ("aircraft6_mean_rmse_change_pct", "Aircraft-targeted"),
    ]
    sub = summary[summary["region"].eq(region)].sort_values("all13_mean_rmse_change_pct")
    mat = sub[[m for m, _ in metrics]].to_numpy()
    vmax = max(3.0, float(np.nanmax(np.abs(mat))) * 1.05)
    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([name for _, name in metrics], fontsize=12, weight="bold")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["display"], fontsize=11)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.2f}%", ha="center", va="center", fontsize=10, weight="bold")
    ax.set_title(f"Full-year 2020: {region}\nmean paired RMSE change", weight="bold", fontsize=14)
    cbar = fig.colorbar(im, ax=ax, pad=0.012)
    cbar.set_label("RMSE % change")
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_region_summary(summary: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("all13_mean_rmse_change_pct", "All 13 variables"),
        ("surface3_mean_rmse_change_pct", "Surface-targeted"),
        ("aircraft6_mean_rmse_change_pct", "Aircraft-targeted"),
    ]
    regions = ["strict_conus", "global", "global_excluding_strict_conus"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True, sharey=True)
    colors = {
        "IGRA + ABO(A1)": "#4c78a8",
        "IGRA + METAR(M5)": "#59a14f",
        "IGRA + ABO(A1) + METAR(M5)": "#9c755f",
        "R+A+S (selected with 2019 data)": "#6f4aa8",
    }
    for ax, region in zip(axes, regions):
        sub = summary[summary["region"].eq(region)].copy()
        x = np.arange(len(metrics))
        width = 0.24
        for k, (_, row) in enumerate(sub.iterrows()):
            vals = [row[m] for m, _ in metrics]
            offset = (k - 1) * width
            ax.bar(x + offset, vals, width=width, label=row["display"], color=colors[row["display"]])
            for xi, val in zip(x + offset, vals):
                ax.text(xi, val + (0.08 if val >= 0 else -0.08), f"{val:+.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in metrics], weight="bold")
        ax.set_title(region, weight="bold")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean paired RMSE change vs R-only (%)")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Full-year 2020 selected interface by region", weight="bold", fontsize=15)
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_per_variable(df: pd.DataFrame, region: str, out: Path) -> None:
    sub = (
        df[df["region"].eq(region)]
        .groupby(["display", "protocol", "variable", "var_short"], as_index=False)
        .agg(mean_rmse_change_pct=("rmse_change_pct_vs_R", "mean"))
    )
    order = [SHORT[v] for v in VARIABLES]
    display = PROTOCOLS[0].display
    g = sub[sub["display"].eq(display)].set_index("var_short").loc[order].reset_index()
    vals = g["mean_rmse_change_pct"].to_numpy()
    fig, ax = plt.subplots(figsize=(12.5, 4.6), constrained_layout=True)
    ax.bar(g["var_short"], vals, color=np.where(vals < 0.0, "#2878b5", "#d95f02"))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean paired RMSE change vs R-only (%)")
    ax.grid(axis="y", alpha=0.20)
    ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Full-year 2020 per-variable RMSE change: {region}", weight="bold", fontsize=15)
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_monthly_trajectory(df: pd.DataFrame, var_set: str, region: str, out: Path) -> None:
    agg = aggregate_set(df[df["region"].eq(region)], var_set)
    monthly = (
        agg.groupby(["display", "protocol", "month"], as_index=False)
        .agg(mean_rmse_change_pct=("rmse_change_pct", "mean"))
    )
    fig, ax = plt.subplots(figsize=(11.8, 5.4), constrained_layout=True)
    for display, g in monthly.groupby("display", sort=False):
        g = g.sort_values("month")
        ax.plot(g["month"], g["mean_rmse_change_pct"], marker="o", linewidth=2.2, label=display)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel(f"{var_set} mean paired RMSE change vs R-only (%)")
    ax.set_title(f"Full-year monthly trajectory: {region}, {var_set}", weight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_protocol_table(region_table: pd.DataFrame, region: str, out: Path) -> None:
    display_cols = ["var_short", "unit", "baseline_rmse_mean", "method_rmse_mean", "mean_rmse_change_pct"]
    table = region_table[display_cols].copy()
    table.columns = ["Variable", "Unit", "R-only RMSE", "R+A+S RMSE", "% change vs R-only"]
    table["R-only RMSE"] = table["R-only RMSE"].map(lambda x: f"{x:.3g}")
    table["R+A+S RMSE"] = table["R+A+S RMSE"].map(lambda x: f"{x:.3g}")
    table["% change vs R-only"] = table["% change vs R-only"].map(lambda x: f"{x:+.2f}%")

    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    ax.axis("off")
    ax.set_title(f"Full-year 2020 result: {region}\nR+A+S interface selected with 2019 data", fontsize=16, weight="bold", pad=14)
    tbl = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.4)
    tbl.scale(1.0, 1.38)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_facecolor("#2b6da8")
            cell.set_text_props(color="white", weight="bold")
        else:
            delta = region_table.iloc[r - 1]["mean_rmse_change_pct"]
            if delta < 0:
                cell.set_facecolor("#dcefe4" if r % 2 else "#b8dfc7")
            elif delta > 0:
                cell.set_facecolor("#f5dddd")
            else:
                cell.set_facecolor("#f1f5f9")
            if c == 4:
                cell.set_text_props(color="#00875a" if delta < 0 else "#cc2f27", weight="bold")
    fig.savefig(out, dpi=240)
    plt.close(fig)


def make_figures(df: pd.DataFrame, summary: pd.DataFrame, per_var: pd.DataFrame) -> None:
    for region in ["strict_conus", "global", "global_excluding_strict_conus"]:
        plot_metric_matrix(summary, region, FIG_DIR / f"full_year_metric_matrix_{region}.png")
        plot_per_variable(df, region, FIG_DIR / f"full_year_per_variable_pct_delta_{region}.png")
        for var_set in ["all13", "surface3", "aircraft6"]:
            plot_monthly_trajectory(df, var_set, region, FIG_DIR / f"full_year_monthly_{var_set}_{region}.png")
        final_table = per_var[
            per_var["protocol"].eq("RplusAplusS_selected_2019") & per_var["region"].eq(region)
        ].sort_values("mean_rmse_change_pct")
        plot_protocol_table(final_table, region, FIG_DIR / f"selected_RAS_result_table_{region}.png")
    plot_region_summary(summary, FIG_DIR / "full_year_region_summary_bars.png")


def write_readme(summary: pd.DataFrame, per_var: pd.DataFrame, meta: dict, outputs: dict[str, Path]) -> None:
    def line_for(region: str, protocol: str) -> str:
        row = summary[summary["region"].eq(region) & summary["protocol"].eq(protocol)].iloc[0]
        return (
            f"- {row['display']}: all 13 variables {row['all13_mean_rmse_change_pct']:+.2f}%, "
            f"surface-targeted {row['surface3_mean_rmse_change_pct']:+.2f}%, "
            f"aircraft-targeted {row['aircraft6_mean_rmse_change_pct']:+.2f}%"
        )

    regions = ["strict_conus", "global", "global_excluding_strict_conus"]
    lines = [
        "# R+A+S Full-Year 2020 Analysis",
        "",
        f"Generated: {pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "This report evaluates the observation interfaces selected with 2019 development data over the 723 matched 12-hour 2020 analysis times.",
        "",
        "Negative percentage changes mean lower RMSE than R-only conditioning.",
        "",
        "## Inputs",
        "",
        f"- R+A+S sample directory: `{RAS_SAMPLES_ROOT}`",
        f"- R-only baseline: `{BASELINE_ROOT}`",
        f"- ERA5 reference state: `{ERA5_ROOT}`",
        f"- Timestep manifest: `{TIMESTEP_MANIFEST}`",
        f"- Number of matched timesteps: {meta['n_timesteps']}",
        "",
        "## Selected configuration",
        "",
        "- `R+A+S (selected with 2019 data)`: radiosonde/profile anchor plus MADIS point/acars aircraft and MADIS METAR surface-station equal-cell interfaces.",
        "- Aircraft parameters: lambda=0.4, std=5e-4, gamma=2e-5.",
        "- Surface-station parameters: lambda=0.4, std=1.25e-4, gamma=4e-5.",
        "- Aircraft reports: retain MADIS data-source codes {0,1,5}; exclude TAMDAR.",
        "",
        "## Main Summary",
        "",
    ]
    for region in regions:
        lines.extend([f"### {region}", ""])
        for protocol in [p.label for p in PROTOCOLS]:
            lines.append(line_for(region, protocol))
        lines.append("")

    selected = summary[summary["protocol"].eq("RplusAplusS_selected_2019")]
    strict = selected[selected["region"].eq("strict_conus")].iloc[0]
    rest = selected[selected["region"].eq("global_excluding_strict_conus")].iloc[0]
    lines.extend([
        "## Interpretation",
        "",
        f"- R+A+S reduces the mean paired RMSE change across all 13 variables within CONUS by {strict['all13_mean_rmse_change_pct']:+.2f}% and across aircraft-targeted variables by {strict['aircraft6_mean_rmse_change_pct']:+.2f}% relative to R-only conditioning.",
        f"- Outside CONUS, the corresponding all-variable change is {rest['all13_mean_rmse_change_pct']:+.3f}%.",
        "",
        "## Outputs",
        "",
        f"- Metrics by timestep/variable: `{TABLE_DIR / 'selected_RAS_metrics_by_timestep_variable.csv'}`",
        f"- Summary by protocol/region: `{TABLE_DIR / 'selected_RAS_summary_by_protocol_region.csv'}`",
        f"- Per-variable summary: `{outputs['per_variable_summary']}`",
        f"- Monthly summary: `{outputs['monthly_summary']}`",
        f"- Figures: `{FIG_DIR}`",
        "",
        "## Sample Counts",
        "",
        "```json",
        json.dumps(meta["sample_counts"], indent=2),
        "```",
        "",
    ])
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df, summary, meta = evaluate()

    metrics_path = TABLE_DIR / "selected_RAS_metrics_by_timestep_variable.csv"
    summary_path = TABLE_DIR / "selected_RAS_summary_by_protocol_region.csv"
    df.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    outputs = write_result_tables(df)
    per_var = pd.read_csv(outputs["per_variable_summary"])
    make_figures(df, summary, per_var)

    manifest = {
        "generated_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "meta": meta,
        "outputs": {
            "tables": str(TABLE_DIR),
            "figures": str(FIG_DIR),
            "metrics_by_timestep_variable": str(metrics_path),
            "summary_by_protocol_region": str(summary_path),
            **{k: str(v) for k, v in outputs.items()},
        },
        "strict_conus_summary": summary[summary["region"].eq("strict_conus")].to_dict(orient="records"),
        "global_excluding_conus_summary": summary[summary["region"].eq("global_excluding_strict_conus")].to_dict(orient="records"),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_readme(summary, per_var, meta, outputs)

    print(json.dumps({
        "out_dir": str(OUT_DIR),
        "summary_table": str(summary_path),
        "figures": str(FIG_DIR),
        "strict_conus_summary": manifest["strict_conus_summary"],
        "global_excluding_conus_summary": manifest["global_excluding_conus_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
