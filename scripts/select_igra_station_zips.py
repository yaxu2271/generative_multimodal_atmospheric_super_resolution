#!/usr/bin/env python3
"""Map observation coordinates in a DJ IGRA pickle to NOAA IGRA station IDs."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np

from independent_year_common import six_hour_index, stratified24


def read_station_list(path: Path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if len(line) < 75:
            continue
        try:
            latitude = float(line[12:20])
            longitude = float(line[21:30])
        except ValueError:
            continue
        if latitude <= -98.0 or longitude <= -998.0:
            continue
        station_id = line[0:11].strip()
        if not re.fullmatch(r"[A-Z0-9]{11}", station_id):
            continue
        rows.append(
            {
                "station_id": station_id,
                "latitude": latitude,
                "longitude": longitude,
                "start_year": int(line[72:76]),
                "end_year": int(line[77:81]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_pkl", type=Path, required=True)
    parser.add_argument("--station_list", type=Path, required=True)
    parser.add_argument("--reference_year", type=int, default=2020)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--tolerance_deg", type=float, default=5.1e-5)
    args = parser.parse_args()

    stations = read_station_list(args.station_list)
    coords = np.asarray([[r["latitude"], r["longitude"]] for r in stations])
    with args.reference_pkl.open("rb") as handle:
        reference = pickle.load(handle)

    observed_coords = set()
    for dt in stratified24(args.reference_year):
        locs_by_channel = reference[six_hour_index(dt)][0][0]
        for locs in locs_by_channel:
            observed_coords.update((round(float(lat), 4), round(float(lon), 4)) for lat, lon in np.asarray(locs))

    selected = set()
    unmatched = []
    ambiguous = []
    for lat, lon in sorted(observed_coords):
        delta = np.maximum(np.abs(coords[:, 0] - lat), np.abs(coords[:, 1] - lon))
        matches = np.flatnonzero(delta <= args.tolerance_deg)
        if not len(matches):
            unmatched.append((lat, lon))
            continue
        if len(matches) > 1:
            ambiguous.append((lat, lon, [stations[i]["station_id"] for i in matches]))
        selected.update(stations[i]["station_id"] for i in matches)

    rows = [r for r in stations if r["station_id"] in selected]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "reference_pkl": str(args.reference_pkl.resolve()),
        "reference_year": args.reference_year,
        "n_unique_observed_coordinates": len(observed_coords),
        "n_selected_station_ids": len(selected),
        "n_unmatched_coordinates": len(unmatched),
        "n_ambiguous_coordinates": len(ambiguous),
        "unmatched_coordinates_first20": unmatched[:20],
        "ambiguous_coordinates_first20": ambiguous[:20],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
