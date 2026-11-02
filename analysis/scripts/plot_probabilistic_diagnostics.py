#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_paths import required_path

PROBABILISTIC_TABLES = required_path("PROBABILISTIC_TABLE_ROOT")
DISPERSION_TABLES = required_path("DISPERSION_TABLE_ROOT")
FIGURES = Path(__file__).resolve().parents[1] / "figures"

GROUP_ORDER = [
    "all_13_variables",
    "surface_targeted_variables",
    "aircraft_targeted_variables",
]
GROUP_LABELS = {
    "all_13_variables": "All 13",
    "surface_targeted_variables": "Surface-targeted",
    "aircraft_targeted_variables": "Aircraft-targeted",
}


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "axes.titleweight": "bold",
        }
    )


def save(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_probabilistic_summary() -> None:
    summary = pd.read_csv(
        PROBABILISTIC_TABLES / "area_weighted_probabilistic_group_summary.csv"
    )
    strict = summary[summary["region"].eq("strict_conus")].set_index(
        "evaluation_group"
    )
    labels = [GROUP_LABELS[group] for group in GROUP_ORDER]
    x = np.arange(len(GROUP_ORDER))

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.85), constrained_layout=True)

    axes[0].bar(
        x - 0.18,
        strict.loc[GROUP_ORDER, "mean_ensemble_mean_rmse_pct_change"],
        0.36,
        color="#2B6EA6",
        label="Ensemble-mean RMSE",
    )
    axes[0].bar(
        x + 0.18,
        strict.loc[GROUP_ORDER, "mean_crps_pct_change"],
        0.36,
        color="#D97720",
        label="CRPS",
    )
    axes[0].axhline(0, color="#444444", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("Change relative to R-only conditioning (%)")
    axes[0].set_title("(a) Distributional skill")
    axes[0].legend(frameon=True, fontsize=9)

    width = 0.36
    axes[1].bar(
        x - width / 2,
        strict.loc[GROUP_ORDER, "R_coverage_05_95"],
        width,
        color="#8AB6D6",
        label="R",
    )
    axes[1].bar(
        x + width / 2,
        strict.loc[GROUP_ORDER, "RAS_coverage_05_95"],
        width,
        color="#6AB89A",
        label="R+A+S",
    )
    axes[1].axhline(
        0.9,
        color="#444444",
        linestyle="--",
        linewidth=1,
        label="Large-ensemble 0.90 reference",
    )
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Empirical 5%-95% coverage")
    axes[1].set_title("(b) Finite-ensemble coverage")
    axes[1].legend(frameon=True, fontsize=8)

    save(figure, "probabilistic_diagnostics_conus")


def plot_dispersion_summary() -> None:
    spread = pd.read_csv(
        DISPERSION_TABLES / "area_weighted_spread_skill_coverage_by_group.csv"
    )
    ranks = pd.read_csv(
        DISPERSION_TABLES / "area_weighted_rank_histogram_by_group.csv"
    )
    labels = [GROUP_LABELS[group] for group in GROUP_ORDER]
    x = np.arange(len(GROUP_ORDER))
    width = 0.34
    colors = {"R": "#2A6FAD", "R+A+S": "#D97706"}
    markers = {"R": "o", "R+A+S": "D"}

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.55), constrained_layout=True)
    for offset, protocol in [(-width / 2, "R"), (width / 2, "R+A+S")]:
        subset = spread[spread["protocol"].eq(protocol)].set_index(
            "evaluation_group"
        )
        axes[0].bar(
            x + offset,
            subset.loc[GROUP_ORDER, "mean_corrected_spread_skill_ratio"],
            width,
            color=colors[protocol],
            edgecolor="white",
            linewidth=0.7,
            label=protocol,
        )
    axes[0].axhline(
        1.0,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="Reference value = 1",
    )
    axes[0].set_xticks(x, labels, rotation=8)
    axes[0].set_ylabel("Corrected spread-skill ratio")
    axes[0].set_title("(a) Dispersion relative to mean error")
    axes[0].set_ylim(0.0, 1.08)
    axes[0].grid(axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.85)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="#C7CDD3")

    all_ranks = ranks[
        ranks["region"].eq("strict_conus")
        & ranks["evaluation_group"].eq("all_13_variables")
    ]
    for protocol in ["R", "R+A+S"]:
        subset = all_ranks[all_ranks["protocol"].eq(protocol)].sort_values("rank")
        axes[1].plot(
            subset["rank"],
            subset["frequency_relative_to_uniform"],
            color=colors[protocol],
            linewidth=2.1,
            marker=markers[protocol],
            markersize=4.6,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=protocol,
        )
    axes[1].axhline(
        1.0,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="Uniform reference",
    )
    axes[1].set_xticks([0, 4, 8, 12, 16])
    axes[1].set_xlabel("Verification rank among 16 members")
    axes[1].set_ylabel("Rank frequency / uniform frequency")
    axes[1].set_title("(b) All-variable rank histogram")
    axes[1].grid(color="#D9DEE3", linewidth=0.7, alpha=0.85)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="#C7CDD3")

    save(figure, "ensemble_dispersion_diagnostics_conus")


def main() -> None:
    configure_plotting()
    plot_probabilistic_summary()
    plot_dispersion_summary()
    print(FIGURES / "probabilistic_diagnostics_conus.pdf")
    print(FIGURES / "ensemble_dispersion_diagnostics_conus.pdf")


if __name__ == "__main__":
    main()
