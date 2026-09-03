#!/usr/bin/env python3
"""Generate the Stage IV 3x3 joint-interaction manifest."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from run_independent_year_2019_protocol import DEFAULT_MANIFEST, load_frozen_manifest, sha256
from run_independent_year_2019_stage3_protocol import DEFAULT_SELECTION, load_selection


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO / "configs/independent_year_2019/stage4_joint3x3_manifest_v1.json"
)

AIRCRAFT_CANDIDATES = [
    {
        "name": "a_old_a1",
        "role": "legacy 2020 A1 center",
        "parameters": {"lambda": 0.1, "std": 5e-4, "gamma": 5e-6},
    },
    {
        "name": "a_2019_best",
        "role": "2019 Stage III-D aircraft all13 and target-group best",
        "parameters": {"lambda": 0.4, "std": 5e-4, "gamma": 2e-5},
    },
    {
        "name": "a_near_tie",
        "role": "2019 near-tied interior aircraft candidate",
        "parameters": {"lambda": 0.2, "std": 2.5e-4, "gamma": 1e-5},
    },
]

SURFACE_CANDIDATES = [
    {
        "name": "s_old_m5",
        "role": "legacy 2020 M5 center and 2019 all13 surface best",
        "parameters": {"lambda": 0.2, "std": 5e-4, "gamma": 1e-5},
    },
    {
        "name": "s_compromise",
        "role": "2019 interior compromise between all13 and surface-targeted candidates",
        "parameters": {"lambda": 0.4, "std": 1.25e-4, "gamma": 4e-5},
    },
    {
        "name": "s_aggressive",
        "role": "2019 surface-targeted boundary best",
        "parameters": {"lambda": 0.8, "std": 6.25e-5, "gamma": 8e-5},
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection-artifact", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base, base_digest = load_frozen_manifest(args.base_manifest)
    _, selection_digest = load_selection(args.selection_artifact)

    protocols: list[dict] = []
    for aircraft, surface in itertools.product(AIRCRAFT_CANDIDATES, SURFACE_CANDIDATES):
        protocols.append(
            {
                "protocol_id": f"joint3x3__{aircraft['name']}__{surface['name']}",
                "stage": "joint_interaction_3x3",
                "aircraft_candidate": aircraft,
                "surface_candidate": surface,
                "aircraft_parameters": aircraft["parameters"],
                "surface_parameters": surface["parameters"],
            }
        )

    manifest = {
        "manifest_version": "stage4_joint3x3_manifest_v1",
        "purpose": (
            "Stage IV joint interaction over three aircraft and three "
            "surface-station numerical-calibration candidates. Selection is "
            "intended to use strict-CONUS all-13 improvement on the frozen "
            "2019 development cases."
        ),
        "base_candidate_manifest": str(args.base_manifest.resolve()),
        "base_candidate_manifest_sha256": base_digest,
        "selection_artifact": str(args.selection_artifact.resolve()),
        "selection_artifact_sha256": selection_digest,
        "fixed_sampling": base["fixed_sampling"],
        "aircraft_candidates": AIRCRAFT_CANDIDATES,
        "surface_candidates": SURFACE_CANDIDATES,
        "selection_metric": "strict_conus_all13_mean_rmse_percent_change_vs_radiosonde_only",
        "protocol_count": len(protocols),
        "protocols": protocols,
    }
    if len(protocols) != 9:
        raise ValueError(f"Expected 9 Stage IV protocols, found {len(protocols)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".sha256").write_text(
        f"{sha256(args.output)}  {args.output.name}\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
