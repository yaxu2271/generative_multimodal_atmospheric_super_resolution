#!/usr/bin/env python3
"""Freeze the complete pre-GPU 2019 development input bundle.

This script is intentionally separate from the data builders and sampler.  It
hashes the already audited inputs, processed observations, frozen selection
rules, and executable code snapshot into one machine-readable provenance file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/depot/rmaulik/data/yangxu")
REPO = ROOT / "repos/observation_interface_independent_year_2019"
REPORT = (
    ROOT
    / "reports/2026/07212026report/20260721__2019_independent_year_protocol_audit"
)
ERA5_ROOT = ROOT / "data_from_DJ_original_NERSC/1.40625deg_from_full_res_1_step_6hr_h5df"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def tree_records(root: Path) -> list[dict]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return [file_record(path) for path in sorted(root.rglob("*")) if path.is_file()]


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def era5_records() -> list[dict]:
    availability = REPORT / "tables/stratified24_2019_local_era5_availability.csv"
    with availability.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24:
        raise ValueError(f"Expected 24 ERA5 development rows, found {len(rows)}")
    records = []
    for row in rows:
        path = Path(row["era5_path"])
        record = file_record(path)
        record.update({"datetime_utc": row["datetime_utc"], "era5_index": int(row["era5_index"])})
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    status = git_text("status", "--porcelain")
    if status:
        raise RuntimeError("Refusing to freeze bundle from a dirty git tree:\n" + status)

    aircraft5 = (
        ROOT
        / "processed/observation_interface_2019/aircraft_madis_abo_13var_2019_strat24_around5_all_sources_v1"
    )
    aircraft25 = (
        ROOT
        / "processed/observation_interface_2019/aircraft_madis_abo_13var_2019_strat24_around25_all_sources_v1"
    )
    metar = (
        ROOT
        / "processed/observation_interface_2019/surface_madis_metar_13var_2019_strat24_strict_conus_v1"
    )
    igra = (
        ROOT
        / "processed/observation_interface_2019/igra_13var_2019_strat24_noaa_v22_rebuild_v1"
    )
    raw_madis = ROOT / "data/observation_interface_2019/raw/madis_stratified24_v1"
    raw_igra = ROOT / "data/observation_interface_2019/raw/igra_v2_por_parity_v1"

    required_control = [
        REPO / "configs/independent_year_2019/candidate_manifest_2019_v1.json",
        REPO / "configs/independent_year_2019/candidate_manifest_2019_v1.sha256",
        REPO / "configs/independent_year_2019/SELECTION_RULE_2019.md",
        REPO / "configs/independent_year_2019/IGRA_PARITY_GATE.md",
        REPO / "configs/independent_year_2019/IGRA_PARITY_GATE_V2.md",
        REPORT / "igra_parity_gate_result.json",
        REPORT / "igra_parity_gate_v2_result.json",
        REPORT / "igra_2020_full_archive_parity_summary.json",
        REPORT / "igra_2020_spatial_parity_25km_summary.json",
        REPORT / "madis_2019_strat24_full_audit_summary.json",
        raw_madis / "download_manifest.csv",
        raw_madis / "download_manifest.json",
        raw_igra / "download_manifest_active_2019_2020.csv",
        raw_igra / "download_summary_active_2019_2020.json",
        igra / "manifest.json",
    ]
    code_files = [
        REPO / "scripts/run_independent_year_2019_protocol.py",
        REPO / "scripts/preprocess_madis_aircraft_13var_npz.py",
        REPO / "scripts/preprocess_madis_metar_13var_npz.py",
        REPO / "scripts/build_igra_from_noaa_por.py",
        REPO / "src/igra_gen/run_aircraft_13var_persistent.py",
        REPO / "src/igra_gen/generating/conditioning_methods.py",
    ]
    checkpoint = (
        ROOT
        / "repos/goes_posterior_sampling_13var/src/results/era5_cond_13/20250913_180217/checkpoints/checkpoint-037129.pt"
    )

    payload = {
        "schema": "observation-interface-independent-year-pre-gpu-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "2019 development only; no 2020 ranking information is permitted",
        "git": {
            "repo": str(REPO),
            "commit": git_text("rev-parse", "HEAD"),
            "branch": git_text("branch", "--show-current"),
            "tree_clean": True,
        },
        "control_files": [file_record(path) for path in required_control],
        "executable_code": [file_record(path) for path in code_files],
        "fixed_prior": {
            "checkpoint": file_record(checkpoint),
            "normalization_mean": file_record(ERA5_ROOT / "normalize_mean.npz"),
            "normalization_std": file_record(ERA5_ROOT / "normalize_std.npz"),
            "latitude": file_record(ERA5_ROOT / "lat.npy"),
            "longitude": file_record(ERA5_ROOT / "lon.npy"),
        },
        "era5_development_cases": era5_records(),
        "processed_observations": {
            "aircraft_around5": tree_records(aircraft5),
            "aircraft_around25": tree_records(aircraft25),
            "metar_strict_conus": tree_records(metar),
            "igra_2019": tree_records(igra),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "git": payload["git"]}, indent=2))


if __name__ == "__main__":
    main()
