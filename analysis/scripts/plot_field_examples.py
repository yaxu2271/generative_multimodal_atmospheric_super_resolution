#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import struct

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from analysis_paths import required_path

ERA5 = required_path("ERA5_ROOT")
TIMESTEP = 278
R_PATH = required_path("FIELD_EXAMPLE_R_SAMPLE")
RAS_PATH = required_path("FIELD_EXAMPLE_RAS_SAMPLE")
SHAPEFILE = required_path("NATURAL_EARTH_SHAPEFILE")
OUT = Path(__file__).resolve().parents[1] / "figures"

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
TARGETS = {
    "t2m": {
        "variable": "2m_temperature",
        "title": "2-m temperature",
        "unit": "K",
        "cmap": "viridis",
    },
    "v500": {
        "variable": "v_component_of_wind_500",
        "title": "500-hPa meridional wind",
        "unit": r"m s$^{-1}$",
        "cmap": "viridis",
    },
}


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )


def read_polygons(path: Path) -> list[list[np.ndarray]]:
    polygons: list[list[np.ndarray]] = []
    with path.open("rb") as handle:
        handle.seek(100)
        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            _, length_words = struct.unpack(">2i", header)
            content = handle.read(length_words * 2)
            if len(content) < 44 or struct.unpack("<i", content[:4])[0] not in (5, 15, 25):
                continue
            n_parts, n_points = struct.unpack("<2i", content[36:44])
            parts_end = 44 + 4 * n_parts
            parts = list(struct.unpack(f"<{n_parts}i", content[44:parts_end])) + [n_points]
            points = np.frombuffer(content[parts_end : parts_end + 16 * n_points], dtype="<f8").reshape(-1, 2)
            polygons.append([points[parts[i] : parts[i + 1]].copy() for i in range(n_parts)])
    return polygons


def draw_boundaries(axis, polygons: list[list[np.ndarray]]) -> None:
    for polygon in polygons:
        for part in polygon:
            if part.size == 0:
                continue
            if part[:, 0].max() < -125 or part[:, 0].min() > -66 or part[:, 1].max() < 24 or part[:, 1].min() > 50:
                continue
            axis.plot(part[:, 0], part[:, 1], color="#303840", linewidth=0.65, alpha=0.8, zorder=5)


def robust_range(fields: list[np.ndarray], symmetric: bool = False) -> tuple[float, float]:
    values = np.concatenate([field[np.isfinite(field)].ravel() for field in fields])
    if symmetric:
        limit = max(float(np.percentile(np.abs(values), 99)), 1e-8)
        return -limit, limit
    lo, hi = np.percentile(values, [1, 99])
    return float(lo), float(hi)


def rmse(a: np.ndarray, b: np.ndarray, lat: np.ndarray) -> float:
    weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, a.shape[1]))
    squared = np.asarray((a - b) ** 2, dtype=np.float64)
    return float(np.sqrt(np.sum(weights * squared) / np.sum(weights)))


def panel(axis, lon, lat, field, title, cmap, vmin, vmax, polygons, norm=None):
    dlon = float(np.median(np.diff(lon)))
    dlat = float(np.median(np.diff(lat)))
    plot_extent = [
        float(lon.min() - dlon / 2.0),
        float(lon.max() + dlon / 2.0),
        float(lat.min() - dlat / 2.0),
        float(lat.max() + dlat / 2.0),
    ]
    image = axis.imshow(
        field,
        origin="lower",
        extent=plot_extent,
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
        rasterized=True,
    )
    axis.set_title(title, fontweight="bold", pad=5)
    axis.set_xlim(plot_extent[0], plot_extent[1])
    axis.set_ylim(plot_extent[2], plot_extent[3])
    axis.set_xticks([-120, -105, -90, -75])
    axis.set_yticks([25, 35, 45])
    axis.set_aspect(1.0 / np.cos(np.deg2rad(37.0)), adjustable="box")
    draw_boundaries(axis, polygons)
    return image


def make_figure(short: str, spec: dict[str, str], r: np.ndarray, ras: np.ndarray, truth_handle, mean, std, iy, ix, lat, lon, polygons):
    channel = VARIABLES.index(spec["variable"])
    r_members = np.asarray(r[:, channel][:, iy][:, :, ix], dtype=np.float32) * std[channel] + mean[channel]
    ras_members = np.asarray(ras[:, channel][:, iy][:, :, ix], dtype=np.float32) * std[channel] + mean[channel]
    r_mean = r_members.mean(axis=0)
    ras_mean = ras_members.mean(axis=0)
    truth = np.asarray(truth_handle["input"][spec["variable"]], dtype=np.float32)[np.ix_(iy, ix)]
    r_error = np.abs(r_mean - truth)
    ras_error = np.abs(ras_mean - truth)
    field_min, field_max = robust_range([r_mean, ras_mean, truth], symmetric=short != "t2m")
    error_min = 0.0
    error_max = max(
        float(np.percentile(np.concatenate([r_error.ravel(), ras_error.ravel()]), 97.5)),
        1e-8,
    )
    error_change = ras_error - r_error
    change_max = max(float(np.percentile(np.abs(error_change), 97.5)), 1e-8)
    r_rmse = rmse(r_mean, truth, lat)
    ras_rmse = rmse(ras_mean, truth, lat)
    change = 100.0 * (ras_rmse - r_rmse) / r_rmse

    figure, axes = plt.subplots(2, 3, figsize=(11.5, 5.8), constrained_layout=True)
    figure.suptitle(
        f"Selected 2020 analysis time over the CONUS domain: {spec['title']}  |  "
        f"Area-weighted RMSE {r_rmse:.2f} to {ras_rmse:.2f} {spec['unit']} ({change:+.1f}%)",
        fontweight="bold",
        fontsize=13,
    )
    top_images = [
        panel(axes[0, 0], lon, lat, r_mean, "(a) R-only ensemble mean", spec["cmap"], field_min, field_max, polygons),
        panel(axes[0, 1], lon, lat, ras_mean, "(b) R+A+S ensemble mean", spec["cmap"], field_min, field_max, polygons),
        panel(axes[0, 2], lon, lat, truth, "(c) ERA5 reference", spec["cmap"], field_min, field_max, polygons),
    ]
    error_norm = Normalize(vmin=error_min, vmax=error_max, clip=True)
    err0 = panel(axes[1, 0], lon, lat, r_error, "(d) R-only absolute error", "RdBu_r", error_min, error_max, polygons, error_norm)
    panel(axes[1, 1], lon, lat, ras_error, "(e) R+A+S absolute error", "RdBu_r", error_min, error_max, polygons, error_norm)
    change_norm = Normalize(vmin=-change_max, vmax=change_max, clip=True)
    change_image = panel(
        axes[1, 2],
        lon,
        lat,
        error_change,
        "(f) Absolute-error change",
        "RdBu_r",
        -change_max,
        change_max,
        polygons,
        change_norm,
    )
    for row in range(2):
        axes[row, 0].set_ylabel("Latitude")
        axes[row, 1].tick_params(labelleft=False)
        axes[row, 2].tick_params(labelleft=False)
    for axis in axes[1, :]:
        axis.set_xlabel("Longitude")
    for axis in axes[0, :]:
        axis.tick_params(labelbottom=False)

    field_bar = figure.colorbar(top_images[0], ax=axes[0, :], shrink=0.84, pad=0.012)
    field_bar.set_label(f"Field ({spec['unit']})")
    error_bar = figure.colorbar(err0, ax=axes[1, :2], shrink=0.84, pad=0.012)
    error_bar.set_label(f"Absolute error ({spec['unit']})")
    change_bar = figure.colorbar(change_image, ax=axes[1, 2], shrink=0.84, pad=0.012)
    change_bar.set_label(f"Absolute-error change ({spec['unit']})")

    stem = f"selected_{short}_absolute_errors_and_change"
    figure.savefig(OUT / f"{stem}.pdf")
    figure.savefig(OUT / f"{stem}.png", dpi=250)
    plt.close(figure)
    print(OUT / f"{stem}.pdf")


def main() -> None:
    configure_plotting()
    OUT.mkdir(parents=True, exist_ok=True)
    means = np.load(ERA5 / "normalize_mean.npz")
    stds = np.load(ERA5 / "normalize_std.npz")
    mean = np.asarray([float(np.asarray(means[v]).reshape(-1)[0]) for v in VARIABLES])
    std = np.asarray([float(np.asarray(stds[v]).reshape(-1)[0]) for v in VARIABLES])
    lat_all = np.load(ERA5 / "lat.npy").astype(float)
    lon_all = np.load(ERA5 / "lon.npy").astype(float)
    lon180 = ((lon_all + 180.0) % 360.0) - 180.0
    iy = np.where((lat_all >= 24.0) & (lat_all <= 50.0))[0]
    ix = np.where((lon180 >= -125.0) & (lon180 <= -66.0))[0]
    ix = ix[np.argsort(lon180[ix])]
    lat = lat_all[iy]
    lon = lon180[ix]
    polygons = read_polygons(SHAPEFILE)
    r = np.load(R_PATH, mmap_mode="r")
    ras = np.load(RAS_PATH, mmap_mode="r")
    with h5py.File(ERA5 / f"test/2020_{TIMESTEP:04d}.h5", "r") as truth_handle:
        for short, spec in TARGETS.items():
            make_figure(short, spec, r, ras, truth_handle, mean, std, iy, ix, lat, lon, polygons)


if __name__ == "__main__":
    main()
