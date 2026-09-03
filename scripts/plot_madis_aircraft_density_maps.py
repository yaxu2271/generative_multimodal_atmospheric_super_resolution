#!/usr/bin/env python3
"""Plot MADIS aircraft candidate point distributions.

This is a diagnostic plotting utility for the aircraft-observation feasibility
branch. It uses the same conservative QCR/physical filters as the obs-vs-ERA5
diagnostic and the same altitude-derived pressure windows.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NATURAL_EARTH_SHP = Path(
    "/depot/rmaulik/data/yangxu/data_aux/natural_earth/ne_110m_admin_0_countries/"
    "ne_110m_admin_0_countries.shp"
)
ERA5_ROOT = Path("/depot/rmaulik/data/yangxu/era5_subset")

STRICT_CONUS = (24.0, 50.0, -125.0, -66.0)
NORTH_AMERICA = (10.0, 70.0, -170.0, -50.0)
GLOBAL = (-90.0, 90.0, -180.0, 180.0)


@dataclass(frozen=True)
class Candidate:
    product: str
    level: str
    kind: str
    lat: np.ndarray
    lon: np.ndarray
    pressure: np.ndarray


def lon_to_180(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def era5_region_points(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> pd.DataFrame:
    lat_grid = np.load(ERA5_ROOT / "lat.npy").astype(float)
    lon_grid = lon_to_180(np.load(ERA5_ROOT / "lon.npy").astype(float))
    lat2, lon2 = np.meshgrid(lat_grid, lon_grid, indexing="ij")
    mask = (
        (lat2 >= lat_min)
        & (lat2 <= lat_max)
        & (lon2 >= lon_min)
        & (lon2 <= lon_max)
    )
    return pd.DataFrame({"lat": lat2[mask], "lon": lon2[mask]})


def read_shp_polygons(path: Path) -> list[list[np.ndarray]]:
    polygons: list[list[np.ndarray]] = []
    if not path.exists():
        return polygons
    with open(path, "rb") as f:
        f.seek(100)
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            _rec_num, content_len_words = struct.unpack(">2i", header)
            content = f.read(content_len_words * 2)
            if len(content) < 44:
                continue
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type not in (5, 15, 25):
                continue
            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts_offset = 44
            points_offset = parts_offset + 4 * num_parts
            if len(content) < points_offset + 16 * num_points:
                continue
            parts = list(struct.unpack(f"<{num_parts}i", content[parts_offset:points_offset]))
            parts.append(num_points)
            pts = np.frombuffer(content[points_offset : points_offset + 16 * num_points], dtype="<f8").reshape(-1, 2)
            polygons.append([pts[parts[i] : parts[i + 1]].copy() for i in range(num_parts)])
    return polygons


def fill_world_land(ax, polygons: list[list[np.ndarray]], *, facecolor: str = "#efe5c8", edgecolor: str = "#3b3b3b") -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for poly in polygons:
        for part in poly:
            if part.size == 0:
                continue
            lon = part[:, 0]
            lat = part[:, 1]
            if lon.max() < xlim[0] or lon.min() > xlim[1] or lat.max() < ylim[0] or lat.min() > ylim[1]:
                continue
            ax.fill(lon, lat, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.55, alpha=0.96, zorder=1)


def decode_to_temp(path: Path) -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="madis_aircraft_density_", suffix=".nc", delete=False)
    tmp_path = Path(tmp.name)
    with gzip.open(path, "rb") as src:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
    tmp.close()
    return tmp_path


def read_var(ds: Any, *names: str) -> np.ndarray | None:
    for name in names:
        if name in ds.variables:
            var = ds.variables[name]
            arr = np.asarray(var[:], dtype=np.float64)
            fill = getattr(var, "_FillValue", None)
            if fill is not None:
                arr = np.where(arr == float(fill), np.nan, arr)
            arr = np.where(arr <= -9990.0, np.nan, arr)
            arr = np.where(arr >= 99990.0, np.nan, arr)
            return arr
    return None


def read_qcr(ds: Any, name: str, target_shape: tuple[int, ...]) -> np.ndarray:
    qname = f"{name}QCR"
    if qname not in ds.variables:
        return np.zeros(target_shape, dtype=np.int64)
    arr = np.asarray(ds.variables[qname][:])
    if arr.shape == target_shape:
        return arr.astype(np.int64)
    if len(target_shape) == 2 and arr.ndim == 1 and arr.shape[0] == target_shape[0]:
        return np.repeat(arr[:, None], target_shape[1], axis=1).astype(np.int64)
    return np.broadcast_to(arr, target_shape).astype(np.int64)


def broadcast_profile_locations(lat: np.ndarray, lon: np.ndarray, target_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    if lat.shape == target_shape:
        return lat, lon
    if len(target_shape) == 2 and lat.ndim == 1 and lat.shape[0] == target_shape[0]:
        return np.repeat(lat[:, None], target_shape[1], axis=1), np.repeat(lon[:, None], target_shape[1], axis=1)
    return np.broadcast_to(lat, target_shape), np.broadcast_to(lon, target_shape)


def read_locations(ds: Any, target_shape: tuple[int, ...]) -> tuple[np.ndarray | None, np.ndarray | None]:
    track_lat = read_var(ds, "trackLat")
    track_lon = read_var(ds, "trackLon")
    if track_lat is not None and track_lon is not None and track_lat.shape == target_shape and track_lon.shape == target_shape:
        return track_lat, track_lon
    lat = read_var(ds, "latitude")
    lon = read_var(ds, "longitude")
    if lat is None or lon is None:
        return None, None
    return broadcast_profile_locations(lat, lon, target_shape)


def pressure_hpa_from_altitude_m(alt_m: np.ndarray) -> np.ndarray:
    alt = np.asarray(alt_m, dtype=np.float64)
    pressure = 1013.25 * np.power(np.maximum(0.0, 1.0 - 2.25577e-5 * alt), 5.25588)
    return np.where(np.isfinite(alt), pressure, np.nan)


def load_candidates(path: Path) -> list[Candidate]:
    from netCDF4 import Dataset

    product = "acarsProfiles" if "acarsProfiles" in str(path) else "acars"
    tmp = decode_to_temp(path)
    try:
        with Dataset(tmp, "r") as ds:
            temp = read_var(ds, "temperature")
            wind_speed = read_var(ds, "windSpeed")
            wind_dir = read_var(ds, "windDir")
            altitude = read_var(ds, "altitude", "GPSaltitude")
            if temp is None or altitude is None:
                return []
            lat2, lon2 = read_locations(ds, temp.shape)
            if lat2 is None or lon2 is None:
                return []
            lon180 = lon_to_180(lon2)
            pressure = pressure_hpa_from_altitude_m(altitude)
            loc_qc = (
                (read_qcr(ds, "latitude", temp.shape) == 0)
                & (read_qcr(ds, "longitude", temp.shape) == 0)
                & (read_qcr(ds, "altitude", temp.shape) == 0)
            )
            temp_qc = (read_qcr(ds, "temperature", temp.shape) == 0) & (temp > 180.0) & (temp < 330.0)
            wind_qc = (
                (read_qcr(ds, "windSpeed", temp.shape) == 0)
                & (read_qcr(ds, "windDir", temp.shape) == 0)
                & np.isfinite(wind_speed)
                & np.isfinite(wind_dir)
                & (wind_speed >= 0.0)
                & (wind_speed < 150.0)
            )
            base = (
                loc_qc
                & np.isfinite(lat2)
                & np.isfinite(lon180)
                & np.isfinite(pressure)
                & (pressure > 100.0)
                & (pressure < 1050.0)
            )
            out: list[Candidate] = []
            for level, lo, hi in [("500", 475.0, 525.0), ("850", 825.0, 875.0)]:
                pwin = (pressure >= lo) & (pressure <= hi)
                for kind, qmask in [("temperature", temp_qc), ("wind", wind_qc)]:
                    mask = base & pwin & qmask
                    out.append(
                        Candidate(
                            product=product,
                            level=level,
                            kind=kind,
                            lat=lat2[mask].astype(float),
                            lon=lon180[mask].astype(float),
                            pressure=pressure[mask].astype(float),
                        )
                    )
            return out
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def concat_candidates(cands: list[Candidate], level: str, kind: str, product: str | None = None) -> pd.DataFrame:
    rows = []
    for c in cands:
        if c.level != level or c.kind != kind:
            continue
        if product is not None and c.product != product:
            continue
        rows.append(pd.DataFrame({"lat": c.lat, "lon": c.lon, "pressure_hpa": c.pressure, "product": c.product}))
    if not rows:
        return pd.DataFrame(columns=["lat", "lon", "pressure_hpa", "product"])
    return pd.concat(rows, ignore_index=True)


def subset_region(df: pd.DataFrame, box: tuple[float, float, float, float]) -> pd.DataFrame:
    lat_min, lat_max, lon_min, lon_max = box
    return df[(df["lat"] >= lat_min) & (df["lat"] <= lat_max) & (df["lon"] >= lon_min) & (df["lon"] <= lon_max)].copy()


def aggregate_to_era5_cells(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["lat", "lon", "count"])
    era5_lat = np.load(ERA5_ROOT / "lat.npy").astype(float)
    era5_lon = np.load(ERA5_ROOT / "lon.npy").astype(float)
    era5_lon180 = lon_to_180(era5_lon)
    # Use nearest regular grid cell. Longitudes are 0..360 in original grid.
    lon360 = np.mod(df["lon"].to_numpy(float), 360.0)
    lat = df["lat"].to_numpy(float)
    dlat = float(np.median(np.diff(era5_lat)))
    dlon = float(np.median(np.diff(era5_lon)))
    iy = np.rint((lat - era5_lat[0]) / dlat).astype(int)
    ix = np.rint(lon360 / dlon).astype(int) % len(era5_lon)
    ok = (iy >= 0) & (iy < len(era5_lat))
    iy = iy[ok]
    ix = ix[ok]
    if len(iy) == 0:
        return pd.DataFrame(columns=["lat", "lon", "count"])
    key, counts = np.unique(np.stack([iy, ix], axis=1), axis=0, return_counts=True)
    return pd.DataFrame({"lat": era5_lat[key[:, 0]], "lon": era5_lon180[key[:, 1]], "count": counts})


def draw_box(ax) -> None:
    lat_min, lat_max, lon_min, lon_max = STRICT_CONUS
    ax.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        color="black",
        linewidth=1.35,
        linestyle="--",
        label="Strict CONUS box",
        zorder=7,
    )


def draw_scatter_map(
    out_path: Path,
    title: str,
    box: tuple[float, float, float, float],
    era5: pd.DataFrame,
    acars: pd.DataFrame,
    profiles: pd.DataFrame,
    polygons: list[list[np.ndarray]],
) -> None:
    lat_min, lat_max, lon_min, lon_max = box
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_facecolor("#d7e8f4")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    fill_world_land(ax, polygons)
    ax.scatter(era5["lon"], era5["lat"], s=12, color="#4d4d4d", alpha=0.34, linewidths=0, label=f"ERA5 grid N={len(era5)}", zorder=3)
    ax.scatter(acars["lon"], acars["lat"], s=18, color="#1f77b4", alpha=0.70, linewidths=0, label=f"MADIS acars N={len(acars)}", zorder=4)
    ax.scatter(profiles["lon"], profiles["lat"], s=18, color="#9467bd", alpha=0.62, linewidths=0, label=f"MADIS acarsProfiles N={len(profiles)}", zorder=5)
    draw_box(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, weight="bold")
    ax.grid(alpha=0.18, color="white")
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def draw_cell_density_map(
    out_path: Path,
    title: str,
    box: tuple[float, float, float, float],
    era5: pd.DataFrame,
    cells: pd.DataFrame,
    polygons: list[list[np.ndarray]],
) -> None:
    lat_min, lat_max, lon_min, lon_max = box
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_facecolor("#d7e8f4")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    fill_world_land(ax, polygons)
    ax.scatter(era5["lon"], era5["lat"], s=11, color="#4d4d4d", alpha=0.25, linewidths=0, label=f"ERA5 grid N={len(era5)}", zorder=3)
    if not cells.empty:
        sizes = np.clip(16 + 4.5 * np.sqrt(cells["count"].to_numpy(float)), 18, 180)
        sc = ax.scatter(
            cells["lon"],
            cells["lat"],
            s=sizes,
            c=cells["count"],
            cmap="viridis",
            alpha=0.82,
            linewidths=0.25,
            edgecolors="white",
            label=f"Covered ERA5 cells N={len(cells)}",
            zorder=4,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.015)
        cbar.set_label("native aircraft points per ERA5 cell")
    draw_box(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, weight="bold")
    ax.grid(alpha=0.18, color="white")
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    polygons = read_shp_polygons(NATURAL_EARTH_SHP)
    all_candidates: list[Candidate] = []
    for path in args.files:
        all_candidates.extend(load_candidates(path))

    rows = []
    for level in ["500", "850"]:
        for kind in ["temperature", "wind"]:
            all_df = concat_candidates(all_candidates, level, kind)
            for region_name, box in [("strict_conus", STRICT_CONUS), ("north_america", NORTH_AMERICA), ("global", GLOBAL)]:
                sub = subset_region(all_df, box)
                cells = aggregate_to_era5_cells(sub)
                rows.append(
                    {
                        "level": level,
                        "kind": kind,
                        "region": region_name,
                        "native_points": int(len(sub)),
                        "covered_era5_cells": int(len(cells)),
                        "mean_points_per_covered_cell": float(len(sub) / len(cells)) if len(cells) else 0.0,
                        "max_points_per_cell": int(cells["count"].max()) if len(cells) else 0,
                    }
                )
    with (args.out_dir / "madis_aircraft_t0000_density_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for level, kind in [("850", "wind"), ("500", "wind"), ("850", "temperature")]:
        all_acars = concat_candidates(all_candidates, level, kind, "acars")
        all_profiles = concat_candidates(all_candidates, level, kind, "acarsProfiles")
        for region_name, box in [("north_america", NORTH_AMERICA), ("strict_conus", STRICT_CONUS), ("global", GLOBAL)]:
            era5 = era5_region_points(*box)
            acars = subset_region(all_acars, box)
            profiles = subset_region(all_profiles, box)
            draw_scatter_map(
                out_path=args.out_dir / f"t0000_madis_aircraft_{kind}_{level}_{region_name}_scatter_worldmap.png",
                title=f"t0000 MADIS aircraft {kind} {level} hPa candidates over {region_name.replace('_', ' ')}",
                box=box,
                era5=era5,
                acars=acars,
                profiles=profiles,
                polygons=polygons,
            )
            combined = pd.concat([acars, profiles], ignore_index=True)
            cells = aggregate_to_era5_cells(combined)
            cells = subset_region(cells, box)
            draw_cell_density_map(
                out_path=args.out_dir / f"t0000_madis_aircraft_{kind}_{level}_{region_name}_era5_cell_density_worldmap.png",
                title=f"t0000 MADIS aircraft {kind} {level} hPa ERA5-cell density over {region_name.replace('_', ' ')}",
                box=box,
                era5=era5,
                cells=cells,
                polygons=polygons,
            )
    print(f"Wrote figures and summary to {args.out_dir}")


if __name__ == "__main__":
    main()
