#!/usr/bin/env python3
"""Generate the Stage III-D directional-boundary manifest."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from run_independent_year_2019_protocol import DEFAULT_MANIFEST, load_frozen_manifest, sha256
from run_independent_year_2019_stage3_protocol import DEFAULT_SELECTION, load_selection
from list_stage3c_protocols import load_manifest as load_stage3c_manifest


REPO = Path(__file__).resolve().parents[1]
DEFAULT_STAGE3C = (
    REPO
    / "configs/independent_year_2019/stage3c_directional_cartesian4_manifest_v1.json"
)
DEFAULT_OUTPUT = (
    REPO
    / "configs/independent_year_2019/stage3d_directional_boundary_manifest_v1.json"
)

GRIDS = {
    "aircraft": {
        "stage": "aircraft_numerical_calibration",
        "parameter_key": "aircraft_parameters",
        "prefix": "aircraft_bound",
        "lambda": [0.2, 0.4, 0.8],
        "std": [0.00025, 0.0005, 0.001],
        "gamma": [0.00001, 0.00002, 0.00004],
    },
    "surface_station": {
        "stage": "metar_numerical_calibration",
        "parameter_key": "metar_parameters",
        "prefix": "surface_bound",
        "lambda": [0.2, 0.4, 0.8],
        "std": [0.0000625, 0.000125, 0.00025],
        "gamma": [0.00002, 0.00004, 0.00008],
    },
}


def value_text(value: float) -> str:
    return format(value, ".12g")


def parameter_tuple(parameters: dict) -> tuple[float, float, float]:
    return (
        float(parameters["lambda"]),
        float(parameters["std"]),
        float(parameters["gamma"]),
    )


def collect_existing(base: dict, stage3c: dict, modality: str, grid: dict) -> set[tuple[float, float, float]]:
    existing: set[tuple[float, float, float]] = set()
    for item in base["protocols"]:
        if item["stage"] == grid["stage"] and grid["parameter_key"] in item:
            existing.add(parameter_tuple(item[grid["parameter_key"]]))
    for item in stage3c["protocols"]:
        if item["stage"] == grid["stage"] and grid["parameter_key"] in item:
            existing.add(parameter_tuple(item[grid["parameter_key"]]))
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection-artifact", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--stage3c-manifest", type=Path, default=DEFAULT_STAGE3C)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base, base_digest = load_frozen_manifest(args.base_manifest)
    _, selection_digest = load_selection(args.selection_artifact)
    stage3c, stage3c_digest = load_stage3c_manifest(args.stage3c_manifest)
    if stage3c["base_candidate_manifest_sha256"] != base_digest:
        raise ValueError("Stage III-C manifest does not match base manifest")
    if stage3c["selection_artifact_sha256"] != selection_digest:
        raise ValueError("Stage III-C manifest does not match selection artifact")

    protocols: list[dict] = []
    reuse: dict[str, dict] = {}
    for modality, grid in GRIDS.items():
        existing = collect_existing(base, stage3c, modality, grid)
        full = list(itertools.product(grid["lambda"], grid["std"], grid["gamma"]))
        missing = [item for item in full if item not in existing]
        reuse[modality] = {
            "full_grid_count": len(full),
            "already_available_count": len(full) - len(missing),
            "new_stage3d_count": len(missing),
        }
        for lambda_value, std_value, gamma_value in missing:
            protocol_id = (
                f"{grid['prefix']}"
                f"__lambda{value_text(lambda_value)}"
                f"__std{value_text(std_value)}"
                f"__gamma{value_text(gamma_value)}"
            )
            protocols.append(
                {
                    "protocol_id": protocol_id,
                    "stage": grid["stage"],
                    grid["parameter_key"]: {
                        "lambda": lambda_value,
                        "std": std_value,
                        "gamma": gamma_value,
                    },
                }
            )

    manifest = {
        "manifest_version": "stage3d_directional_boundary_manifest_v1",
        "purpose": (
            "Directional one-layer boundary expansion after Stage III-C using "
            "only the prespecified 2019 development cases."
        ),
        "base_candidate_manifest": str(args.base_manifest.resolve()),
        "base_candidate_manifest_sha256": base_digest,
        "selection_artifact": str(args.selection_artifact.resolve()),
        "selection_artifact_sha256": selection_digest,
        "stage3c_manifest": str(args.stage3c_manifest.resolve()),
        "stage3c_manifest_sha256": stage3c_digest,
        "fixed_sampling": base["fixed_sampling"],
        "grids": GRIDS,
        "reuse_accounting": reuse,
        "new_protocol_count": len(protocols),
        "protocols": protocols,
    }
    expected = 34
    if len(protocols) != expected:
        raise ValueError(f"Expected {expected} new protocols, found {len(protocols)}")
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksum = args.output.with_suffix(".sha256")
    checksum.write_text(f"{sha256(args.output)}  {args.output.name}\n")
    print(args.output)


if __name__ == "__main__":
    main()
