#!/usr/bin/env python3
"""Audit IGRA pickle parity with one-to-one geodesic coordinate matching."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_igra_from_noaa_por import VARIABLES
from independent_year_common import six_hour_index, stratified24


EARTH_RADIUS_KM = 6371.0


def unit_sphere(locations: np.ndarray) -> np.ndarray:
    locations = np.asarray(locations, dtype=np.float64)
    lat = np.deg2rad(locations[:, 0])
    lon = np.deg2rad(locations[:, 1])
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def chord_to_km(chord: float) -> float:
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))


def greedy_one_to_one_matches(reference: np.ndarray, rebuilt: np.ndarray, max_distance_km: float):
    """Return deterministic short-edge one-to-one spatial matches."""
    if len(reference) == 0 or len(rebuilt) == 0:
        return []
    ref_xyz = unit_sphere(reference)
    new_xyz = unit_sphere(rebuilt)
    radius = 2.0 * np.sin(max_distance_km / (2.0 * EARTH_RADIUS_KM))
    tree = cKDTree(new_xyz)
    edges = []
    for ref_idx, neighbors in enumerate(tree.query_ball_point(ref_xyz, r=radius)):
        for new_idx in neighbors:
            chord = float(np.linalg.norm(ref_xyz[ref_idx] - new_xyz[new_idx]))
            edges.append((chord, ref_idx, int(new_idx)))
    edges.sort()
    used_ref: set[int] = set()
    used_new: set[int] = set()
    matches = []
    for chord, ref_idx, new_idx in edges:
        if ref_idx in used_ref or new_idx in used_new:
            continue
        used_ref.add(ref_idx)
        used_new.add(new_idx)
        matches.append((ref_idx, new_idx, chord_to_km(chord)))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-pkl", type=Path, required=True)
    parser.add_argument("--rebuilt-pkl", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--max-distance-km", type=float, default=25.0)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    with args.reference_pkl.open("rb") as handle:
        reference = pickle.load(handle)
    with args.rebuilt_pkl.open("rb") as handle:
        rebuilt = pickle.load(handle)

    rows = []
    all_distances: list[float] = []
    all_differences: list[float] = []
    total_reference = 0
    total_rebuilt = 0
    total_matches = 0
    for dt in stratified24(args.year):
        index = six_hour_index(dt)
        ref_locations, ref_values = reference[index][0][0], reference[index][1][0]
        new_locations, new_values = rebuilt[index][0][0], rebuilt[index][1][0]
        for channel, variable in enumerate(VARIABLES):
            ref_locs = np.asarray(ref_locations[channel], dtype=np.float64)
            ref_vals = np.asarray(ref_values[channel], dtype=np.float64)
            new_locs = np.asarray(new_locations[channel], dtype=np.float64)
            new_vals = np.asarray(new_values[channel], dtype=np.float64)
            matches = greedy_one_to_one_matches(ref_locs, new_locs, args.max_distance_km)
            distances = np.asarray([match[2] for match in matches], dtype=np.float64)
            differences = np.asarray(
                [abs(ref_vals[ref_idx] - new_vals[new_idx]) for ref_idx, new_idx, _ in matches],
                dtype=np.float64,
            )
            total_reference += len(ref_vals)
            total_rebuilt += len(new_vals)
            total_matches += len(matches)
            all_distances.extend(distances.tolist())
            all_differences.extend(differences.tolist())
            rows.append(
                {
                    "datetime_utc": dt.isoformat() + "Z",
                    "variable": variable,
                    "n_reference": len(ref_vals),
                    "n_rebuilt": len(new_vals),
                    "n_spatial_matches": len(matches),
                    "reference_coverage": len(matches) / max(1, len(ref_vals)),
                    "distance_km_median": float(np.median(distances)) if len(distances) else "",
                    "distance_km_p95": float(np.quantile(distances, 0.95)) if len(distances) else "",
                    "normalized_abs_diff_median": float(np.median(differences)) if len(differences) else "",
                    "normalized_abs_diff_p95": float(np.quantile(differences, 0.95)) if len(differences) else "",
                    "fraction_within_1e-6": float(np.mean(differences <= 1e-6)) if len(differences) else "",
                }
            )

    distances = np.asarray(all_distances, dtype=np.float64)
    differences = np.asarray(all_differences, dtype=np.float64)
    summary = {
        "matching": "deterministic greedy one-to-one geodesic matching",
        "max_distance_km": args.max_distance_km,
        "reference_pkl": str(args.reference_pkl.resolve()),
        "rebuilt_pkl": str(args.rebuilt_pkl.resolve()),
        "year": args.year,
        "n_reference_values": total_reference,
        "n_rebuilt_values": total_rebuilt,
        "n_matched_values": total_matches,
        "reference_pair_coverage": total_matches / max(1, total_reference),
        "distance_km_median": float(np.median(distances)) if len(distances) else None,
        "distance_km_p95": float(np.quantile(distances, 0.95)) if len(distances) else None,
        "matched_distance_distribution_fraction": {
            str(threshold): float(np.mean(distances <= threshold)) if len(distances) else None
            for threshold in (0.02, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
        },
        "reference_coverage_within_km": {
            str(threshold): float(np.sum(distances <= threshold) / max(1, total_reference))
            for threshold in (0.02, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
        },
        "normalized_abs_diff_mean": float(np.mean(differences)) if len(differences) else None,
        "normalized_abs_diff_median": float(np.median(differences)) if len(differences) else None,
        "normalized_abs_diff_p95": float(np.quantile(differences, 0.95)) if len(differences) else None,
        "fraction_within_1e-6": float(np.mean(differences <= 1e-6)) if len(differences) else None,
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
