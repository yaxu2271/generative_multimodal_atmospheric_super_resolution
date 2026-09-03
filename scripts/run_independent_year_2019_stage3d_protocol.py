#!/usr/bin/env python3
"""Run one protocol from the frozen 2019 Stage III-D boundary manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from igra_gen import run_aircraft_13var_persistent as production  # noqa: E402
from list_stage3d_protocols import (  # noqa: E402
    DEFAULT_MANIFEST as DEFAULT_STAGE3D_MANIFEST,
    load_manifest as load_stage3d_manifest,
)
from run_independent_year_2019_protocol import (  # noqa: E402
    AIRCRAFT_ROOTS,
    DEFAULT_IGRA,
    DEFAULT_MANIFEST,
    DEFAULT_METAR,
    load_frozen_manifest,
)
from run_independent_year_2019_stage3_protocol import (  # noqa: E402
    DEFAULT_SELECTION,
    load_selection,
    resolve_stage3,
    validate_inputs,
)


def find_protocol(manifest: dict, protocol_id: str) -> dict:
    matches = [
        item for item in manifest["protocols"] if item["protocol_id"] == protocol_id
    ]
    if len(matches) != 1:
        raise KeyError(f"Expected one Stage III-D protocol for {protocol_id!r}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection-artifact", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--stage3d-manifest", type=Path, default=DEFAULT_STAGE3D_MANIFEST)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--igra-pkl", type=Path, default=DEFAULT_IGRA)
    parser.add_argument("--metar-root", type=Path, default=DEFAULT_METAR)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base, base_digest = load_frozen_manifest(args.candidate_manifest)
    selection, selection_digest = load_selection(args.selection_artifact)
    stage3d, stage3d_digest = load_stage3d_manifest(args.stage3d_manifest)
    if stage3d["base_candidate_manifest_sha256"] != base_digest:
        raise ValueError("Stage III-D manifest does not match base manifest")
    if stage3d["selection_artifact_sha256"] != selection_digest:
        raise ValueError("Stage III-D manifest does not match selection artifact")

    protocol = find_protocol(stage3d, args.protocol_id)
    spec, modality_params = resolve_stage3(protocol, base, selection)
    timesteps = validate_inputs(
        protocol, base, selection, args.igra_pkl, args.metar_root
    )
    if args.smoke:
        timesteps = [timesteps[0]]

    protocol_root = args.run_root / "protocols" / args.protocol_id
    sampling = dict(base["fixed_sampling"])
    if args.smoke:
        sampling.update({"ensemble": 1, "steps": 5, "mode": "stage3d_smoke"})
    else:
        sampling["mode"] = "frozen_2019_stage3d_directional_boundary"
    resolved = {
        "base_candidate_manifest": str(args.candidate_manifest.resolve()),
        "base_candidate_manifest_sha256": base_digest,
        "selection_artifact": str(args.selection_artifact.resolve()),
        "selection_artifact_sha256": selection_digest,
        "stage3d_manifest": str(args.stage3d_manifest.resolve()),
        "stage3d_manifest_sha256": stage3d_digest,
        "protocol": protocol,
        "production_spec": spec,
        "modality_parameters": modality_params,
        "timesteps": timesteps,
        "calendar_year": 2019,
        "era5_root": production.ERA5_ROOT,
        "era5_split": "val",
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
