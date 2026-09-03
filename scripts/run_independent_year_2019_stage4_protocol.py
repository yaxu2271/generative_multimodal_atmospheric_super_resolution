#!/usr/bin/env python3
"""Run one Stage IV 3x3 joint-interaction protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from igra_gen import run_aircraft_13var_persistent as production  # noqa: E402
from list_stage4_protocols import (  # noqa: E402
    DEFAULT_MANIFEST as DEFAULT_STAGE4_MANIFEST,
    load_manifest as load_stage4_manifest,
)
from run_independent_year_2019_protocol import (  # noqa: E402
    AIRCRAFT_OPERATORS,
    AIRCRAFT_ROOTS,
    DEFAULT_IGRA,
    DEFAULT_MANIFEST,
    DEFAULT_METAR,
    METAR_OPERATORS,
    load_frozen_manifest,
    six_hour_index_text,
)
from run_independent_year_2019_stage3_protocol import (  # noqa: E402
    DEFAULT_SELECTION,
    load_selection,
)


def find_protocol(manifest: dict, protocol_id: str) -> dict:
    matches = [item for item in manifest["protocols"] if item["protocol_id"] == protocol_id]
    if len(matches) != 1:
        raise KeyError(f"Expected one Stage IV protocol for {protocol_id!r}")
    return matches[0]


def resolve_stage4(protocol: dict, base: dict, selection: dict) -> tuple[dict, dict]:
    if protocol["stage"] != "joint_interaction_3x3":
        raise ValueError(f"Stage IV runner refuses stage={protocol['stage']!r}")

    aircraft_selection = selection["aircraft"]
    surface_selection = selection["surface_station"]
    if aircraft_selection["source_codes"] != base["aircraft_source_policies"]["legacy_keep015"]:
        raise ValueError("Selection artifact source codes do not match legacy_keep015")

    aircraft_operator = aircraft_selection["operator"]
    aircraft_obs_space, aircraft_settings = AIRCRAFT_OPERATORS[aircraft_operator]
    if aircraft_obs_space != "aircraft_superob_grid":
        raise ValueError(f"Stage IV expects aircraft grid operator, got {aircraft_obs_space}")
    surface_operator = surface_selection["operator"]
    surface_obs_space, surface_settings = METAR_OPERATORS[surface_operator]
    if surface_obs_space != "surface_superob_grid":
        raise ValueError(f"Stage IV expects surface grid operator, got {surface_obs_space}")

    window = aircraft_selection["pressure_window"]
    spec = {
        "use_igra": True,
        "obs_mode": f"aircraft_{window}",
        "obs_modality": "aircraft_surface",
        "obs_space": "aircraft_surface_superob_grid",
        "aircraft_data_sources": aircraft_selection["source_codes"],
        "aircraft_spatial_support": base["fixed_sampling"]["support"],
        "aircraft_pressure_window_name": window,
        "aircraft_source_policy_name": aircraft_selection["source_policy_name"],
        "surface_variables": production.SURFACE_METAR_VARIABLES,
        "surface_spatial_support": base["fixed_sampling"]["support"],
        "description": (
            "2019 Stage IV: joint profile + aircraft + surface-station "
            "interaction using frozen structural interfaces and 3x3 numerical "
            f"candidate pair {protocol['protocol_id']}."
        ),
        **aircraft_settings,
        **surface_settings,
    }
    params = {
        "aircraft": protocol["aircraft_parameters"],
        "surface": protocol["surface_parameters"],
    }
    return spec, params


def validate_inputs(base: dict, selection: dict, igra_pkl: Path, metar_root: Path) -> list[int]:
    timesteps = [six_hour_index_text(text) for text in base["development_timestamps"]]
    missing: list[str] = []
    if not igra_pkl.is_file():
        missing.append(str(igra_pkl))

    aircraft_root = AIRCRAFT_ROOTS[selection["aircraft"]["pressure_window"]]
    for timestep in timesteps:
        aircraft_path = aircraft_root / "obs" / f"madis_aircraft_13var_t{timestep:04d}.npz"
        surface_path = metar_root / "obs" / f"madis_metar_surface_13var_t{timestep:04d}.npz"
        era5_path = Path(production.ERA5_ROOT) / "val" / f"2019_{timestep:04d}.h5"
        for path in [aircraft_path, surface_path, era5_path]:
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required 2019 inputs:\n" + "\n".join(missing))
    return timesteps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection-artifact", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--stage4-manifest", type=Path, default=DEFAULT_STAGE4_MANIFEST)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--igra-pkl", type=Path, default=DEFAULT_IGRA)
    parser.add_argument("--metar-root", type=Path, default=DEFAULT_METAR)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base, base_digest = load_frozen_manifest(args.candidate_manifest)
    selection, selection_digest = load_selection(args.selection_artifact)
    stage4, stage4_digest = load_stage4_manifest(args.stage4_manifest)
    if stage4["base_candidate_manifest_sha256"] != base_digest:
        raise ValueError("Stage IV manifest does not match base manifest")
    if stage4["selection_artifact_sha256"] != selection_digest:
        raise ValueError("Stage IV manifest does not match selection artifact")

    protocol = find_protocol(stage4, args.protocol_id)
    spec, modality_params = resolve_stage4(protocol, base, selection)
    timesteps = validate_inputs(base, selection, args.igra_pkl, args.metar_root)
    if args.smoke:
        timesteps = [timesteps[0]]

    protocol_root = args.run_root / "protocols" / args.protocol_id
    sampling = dict(base["fixed_sampling"])
    if args.smoke:
        sampling.update({"ensemble": 1, "steps": 5, "mode": "stage4_smoke"})
    else:
        sampling["mode"] = "frozen_2019_stage4_joint3x3"
    resolved = {
        "base_candidate_manifest": str(args.candidate_manifest.resolve()),
        "base_candidate_manifest_sha256": base_digest,
        "selection_artifact": str(args.selection_artifact.resolve()),
        "selection_artifact_sha256": selection_digest,
        "stage4_manifest": str(args.stage4_manifest.resolve()),
        "stage4_manifest_sha256": stage4_digest,
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
    aircraft = modality_params["aircraft"]
    surface = modality_params["surface"]
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
