#!/usr/bin/env python3
"""Run one frozen 2019 Stage III-B outer-guard protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path("/depot/rmaulik/data/yangxu")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from igra_gen import run_aircraft_13var_persistent as production  # noqa: E402
from list_stage3b_protocols import (  # noqa: E402
    DEFAULT_MANIFEST as DEFAULT_GUARD_MANIFEST,
    load_manifest as load_guard_manifest,
)
from run_independent_year_2019_protocol import (  # noqa: E402
    AIRCRAFT_ROOTS,
    DEFAULT_IGRA,
    DEFAULT_MANIFEST,
    DEFAULT_METAR,
    load_frozen_manifest,
    six_hour_index_text,
)
from run_independent_year_2019_stage3_protocol import (  # noqa: E402
    DEFAULT_SELECTION,
    load_selection,
    resolve_stage3,
    validate_inputs,
)


def find_guard(manifest: dict, protocol_id: str) -> dict:
    matches = [
        item for item in manifest["protocols"] if item["protocol_id"] == protocol_id
    ]
    if len(matches) != 1:
        raise KeyError(f"Expected one outer-guard protocol for {protocol_id!r}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection-artifact", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--outer-guard-manifest", type=Path, default=DEFAULT_GUARD_MANIFEST
    )
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--igra-pkl", type=Path, default=DEFAULT_IGRA)
    parser.add_argument("--metar-root", type=Path, default=DEFAULT_METAR)
    parser.add_argument("--timesteps", default="manifest")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base, base_digest = load_frozen_manifest(args.candidate_manifest)
    selection, selection_digest = load_selection(args.selection_artifact)
    guard_manifest, guard_digest = load_guard_manifest(args.outer_guard_manifest)
    if guard_manifest["base_candidate_manifest_sha256"] != base_digest:
        raise ValueError("Outer-guard manifest does not match base candidate manifest")
    if guard_manifest["selection_artifact_sha256"] != selection_digest:
        raise ValueError("Outer-guard manifest does not match selection artifact")

    protocol = find_guard(guard_manifest, args.protocol_id)
    spec, modality_params = resolve_stage3(protocol, base, selection)
    manifest_timesteps = validate_inputs(
        protocol, base, selection, args.igra_pkl, args.metar_root
    )
    if args.smoke:
        timesteps = [manifest_timesteps[0]]
    elif args.timesteps == "manifest":
        timesteps = manifest_timesteps
    else:
        timesteps = production.parse_timesteps(args.timesteps)
        unexpected = sorted(set(timesteps) - set(manifest_timesteps))
        if unexpected:
            raise ValueError(
                f"Requested timesteps outside frozen 2019 development set: {unexpected}"
            )

    protocol_root = args.run_root / "protocols" / args.protocol_id
    sampling = dict(base["fixed_sampling"])
    if args.smoke:
        sampling.update({"ensemble": 1, "steps": 5, "mode": "stage3b_code_path_smoke"})
    else:
        sampling["mode"] = "frozen_2019_stage3b_outer_guard"
    resolved = {
        "base_candidate_manifest": str(args.candidate_manifest.resolve()),
        "base_candidate_manifest_sha256": base_digest,
        "selection_artifact": str(args.selection_artifact.resolve()),
        "selection_artifact_sha256": selection_digest,
        "outer_guard_manifest": str(args.outer_guard_manifest.resolve()),
        "outer_guard_manifest_sha256": guard_digest,
        "protocol": protocol,
        "production_spec": spec,
        "modality_parameters": modality_params,
        "timesteps": timesteps,
        "igra_pkl": str(args.igra_pkl.resolve()),
        "metar_root": str(args.metar_root.resolve()),
        "aircraft_roots": {
            key: str(value.resolve()) for key, value in AIRCRAFT_ROOTS.items()
        },
        "era5_root": production.ERA5_ROOT,
        "era5_split": "val",
        "calendar_year": 2019,
        "checkpoint": production.DEFAULT_CHECKPOINT,
        "effective_sampling": sampling,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return

    protocol_root.mkdir(parents=True, exist_ok=True)
    (protocol_root / "resolved_protocol.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n"
    )
    production.EXPERIMENTS[args.protocol_id] = spec

    fixed_igra = base["fixed_igra"]
    aircraft = modality_params.get(
        "aircraft", {"lambda": 0.1, "std": 5e-4, "gamma": 5e-6}
    )
    surface = modality_params.get(
        "surface", {"lambda": 0.2, "std": 5e-4, "gamma": 1e-5}
    )
    runner = production.PersistentFullPoolRunner(
        output_root=str(protocol_root),
        ens=int(sampling["ensemble"]),
        seed=int(sampling["seed"]),
        num_steps=int(sampling["steps"]),
        igra_pkl=str(args.igra_pkl),
        aircraft_around5_root=str(AIRCRAFT_ROOTS["around5"]),
        aircraft_around25_root=str(AIRCRAFT_ROOTS["around25"]),
        surface_metar_root=str(args.metar_root),
        likelihood_mode="multimodal",
        std_igra=float(fixed_igra["std"]),
        gamma_igra=float(fixed_igra["gamma"]),
        lambda_igra=float(fixed_igra["lambda"]),
        std_aircraft=float(aircraft["std"]),
        gamma_aircraft=float(aircraft["gamma"]),
        lambda_aircraft=float(aircraft["lambda"]),
        std_surface=float(surface["std"]),
        gamma_surface=float(surface["gamma"]),
        lambda_surface=float(surface["lambda"]),
        era5_split="val",
        calendar_year=2019,
    )
    runner.run_many([args.protocol_id], timesteps)


if __name__ == "__main__":
    main()
