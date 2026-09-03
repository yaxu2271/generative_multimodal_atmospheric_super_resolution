#!/usr/bin/env python3
"""Generate the frozen missing-point manifest for the 2019 4 x 4 x 4 grids."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from run_independent_year_2019_protocol import DEFAULT_MANIFEST, load_frozen_manifest
from run_independent_year_2019_stage3_protocol import (
    DEFAULT_SELECTION,
    load_selection,
)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO
    / "configs/independent_year_2019/stage3c_directional_cartesian4_manifest_v1.json"
)

GRIDS = {
    "aircraft": {
        "stage": "aircraft_numerical_calibration",
        "parameter_key": "aircraft_parameters",
        "lambda": [0.05, 0.1, 0.2, 0.4],
        "std": [0.000125, 0.00025, 0.0005, 0.001],
        "gamma": [0.000002, 0.000005, 0.00001, 0.00002],
    },
    "surface_station": {
        "stage": "metar_numerical_calibration",
        "parameter_key": "metar_parameters",
        "lambda": [0.05, 0.1, 0.2, 0.4],
        "std": [0.000125, 0.00025, 0.0005, 0.001],
        "gamma": [0.000005, 0.00001, 0.00002, 0.00004],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection-artifact", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base, base_digest = load_frozen_manifest(args.base_manifest)
    _, selection_digest = load_selection(args.selection_artifact)
    protocols: list[dict] = []
    reuse: dict[str, dict] = {}

    for modality, grid in GRIDS.items():
        existing = {
            parameter_tuple(item[grid["parameter_key"]])
            for item in base["protocols"]
            if item["stage"] == grid["stage"]
        }
        full = list(itertools.product(grid["lambda"], grid["std"], grid["gamma"]))
        missing = [item for item in full if item not in existing]
        if len(full) != 64 or len(existing) != 27 or len(missing) != 37:
            raise ValueError(
                f"Unexpected {modality} counts: full={len(full)}, "
                f"existing={len(existing)}, missing={len(missing)}"
            )
        reuse[modality] = {
            "full_grid_count": len(full),
            "reused_stage3_count": len(existing),
            "new_stage3c_count": len(missing),
        }
        prefix = "aircraft" if modality == "aircraft" else "surface"
        for lambda_value, std_value, gamma_value in missing:
            protocol_id = (
                f"{prefix}_cart4"
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
        "manifest_version": "stage3c_directional_cartesian4_manifest_v1",
        "purpose": (
            "Missing full-quality points required to complete one directional "
            "4 x 4 x 4 numerical-calibration cube per modality using only 2019."
        ),
        "base_candidate_manifest": str(args.base_manifest.resolve()),
        "base_candidate_manifest_sha256": base_digest,
        "selection_artifact": str(args.selection_artifact.resolve()),
        "selection_artifact_sha256": selection_digest,
        "fixed_sampling": base["fixed_sampling"],
        "grids": GRIDS,
        "reuse_accounting": reuse,
        "new_protocol_count": len(protocols),
        "protocols": protocols,
    }
    if len(protocols) != 74:
        raise ValueError(f"Expected 74 new protocols, found {len(protocols)}")
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
