#!/usr/bin/env python3
"""Run one frozen 2019 Stage III numerical-calibration protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path("/depot/rmaulik/data/yangxu")
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts"))

from igra_gen import run_aircraft_13var_persistent as production  # noqa: E402
from run_independent_year_2019_protocol import (  # noqa: E402
    AIRCRAFT_OPERATORS,
    AIRCRAFT_ROOTS,
    DEFAULT_IGRA,
    DEFAULT_MANIFEST,
    DEFAULT_METAR,
    METAR_OPERATORS,
    find_protocol,
    load_frozen_manifest,
    sha256,
    six_hour_index_text,
)


DEFAULT_SELECTION = (
    REPO
    / "configs/independent_year_2019/stage3_selection_keep015_v1.json"
)
ALLOWED_STAGES = {"aircraft_numerical_calibration", "metar_numerical_calibration"}


def load_selection(path: Path) -> tuple[dict, str]:
    digest = sha256(path)
    checksum = path.with_suffix(".sha256")
    if not checksum.is_file():
        raise FileNotFoundError(f"Missing selection-artifact checksum: {checksum}")
    expected = checksum.read_text().split()[0]
    if digest != expected:
        raise ValueError(
            f"Selection-artifact checksum mismatch: expected {expected}, found {digest}"
        )
    selection = json.loads(path.read_text())
    if set(selection["allowed_stages"]) != ALLOWED_STAGES:
        raise ValueError("Selection artifact does not authorize exactly the Stage III stages")
    return selection, digest


def resolve_stage3(protocol: dict, manifest: dict, selection: dict) -> tuple[dict, dict]:
    stage = protocol["stage"]
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"Stage III runner refuses stage={stage!r}")

    if stage == "aircraft_numerical_calibration":
        selected = selection["aircraft"]
        if selected["source_codes"] != manifest["aircraft_source_policies"]["legacy_keep015"]:
            raise ValueError("Selection artifact source codes do not match legacy_keep015")
        operator = selected["operator"]
        obs_space, operator_settings = AIRCRAFT_OPERATORS[operator]
        window = selected["pressure_window"]
        spec = {
            "use_igra": True,
            "obs_mode": f"aircraft_{window}",
            "obs_modality": "aircraft",
            "obs_space": obs_space,
            "aircraft_data_sources": selected["source_codes"],
            "aircraft_spatial_support": manifest["fixed_sampling"]["support"],
            "aircraft_pressure_window_name": window,
            "aircraft_source_policy_name": selected["source_policy_name"],
            "description": (
                "2019 Stage III: IGRA plus aircraft numerical calibration; "
                f"source_policy={selected['source_policy_name']}; "
                f"pressure_window={window}; operator={operator}."
            ),
            **operator_settings,
        }
        return spec, {"aircraft": protocol["aircraft_parameters"]}

    selected = selection["surface_station"]
    operator = selected["operator"]
    obs_space, operator_settings = METAR_OPERATORS[operator]
    spec = {
        "use_igra": True,
        "obs_mode": "surface_metar_strat24",
        "obs_modality": "surface",
        "obs_space": obs_space,
        "surface_variables": production.SURFACE_METAR_VARIABLES,
        "surface_spatial_support": manifest["fixed_sampling"]["support"],
        "description": (
            "2019 Stage III: IGRA plus surface-station numerical calibration; "
            f"operator={operator}."
        ),
        **operator_settings,
    }
    return spec, {"surface": protocol["metar_parameters"]}


def validate_inputs(
    protocol: dict,
    manifest: dict,
    selection: dict,
    igra_pkl: Path,
    metar_root: Path,
) -> list[int]:
    timesteps = [six_hour_index_text(text) for text in manifest["development_timestamps"]]
    missing: list[str] = []
    if not igra_pkl.is_file():
        missing.append(str(igra_pkl))
    if protocol["stage"] == "aircraft_numerical_calibration":
        root = AIRCRAFT_ROOTS[selection["aircraft"]["pressure_window"]]
        for timestep in timesteps:
            path = root / "obs" / f"madis_aircraft_13var_t{timestep:04d}.npz"
            if not path.is_file():
                missing.append(str(path))
    elif protocol["stage"] == "metar_numerical_calibration":
        for timestep in timesteps:
            path = metar_root / "obs" / f"madis_metar_surface_13var_t{timestep:04d}.npz"
            if not path.is_file():
                missing.append(str(path))
    else:
        raise ValueError(f"Stage III runner refuses stage={protocol['stage']!r}")

    era5_val = Path(production.ERA5_ROOT) / "val"
    for timestep in timesteps:
        path = era5_val / f"2019_{timestep:04d}.h5"
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required 2019 inputs:\n" + "\n".join(missing))
    return timesteps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection-artifact", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--igra-pkl", type=Path, default=DEFAULT_IGRA)
    parser.add_argument("--metar-root", type=Path, default=DEFAULT_METAR)
    parser.add_argument("--timesteps", default="manifest")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest, manifest_digest = load_frozen_manifest(args.candidate_manifest)
    selection, selection_digest = load_selection(args.selection_artifact)
    protocol = find_protocol(manifest, args.protocol_id)
    spec, modality_params = resolve_stage3(protocol, manifest, selection)
    manifest_timesteps = validate_inputs(
        protocol, manifest, selection, args.igra_pkl, args.metar_root
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
    sampling = dict(manifest["fixed_sampling"])
    if args.smoke:
        sampling.update({"ensemble": 1, "steps": 5, "mode": "stage3_code_path_smoke"})
    else:
        sampling["mode"] = "frozen_2019_stage3_numerical_calibration"
    resolved = {
        "candidate_manifest": str(args.candidate_manifest.resolve()),
        "candidate_manifest_sha256": manifest_digest,
        "selection_artifact": str(args.selection_artifact.resolve()),
        "selection_artifact_sha256": selection_digest,
        "protocol": protocol,
        "production_spec": spec,
        "modality_parameters": modality_params,
        "timesteps": timesteps,
        "igra_pkl": str(args.igra_pkl.resolve()),
        "metar_root": str(args.metar_root.resolve()),
        "aircraft_roots": {key: str(value.resolve()) for key, value in AIRCRAFT_ROOTS.items()},
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

    fixed_igra = manifest["fixed_igra"]
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
