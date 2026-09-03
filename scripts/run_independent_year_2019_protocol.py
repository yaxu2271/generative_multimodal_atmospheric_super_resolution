#!/usr/bin/env python3
"""Run one frozen 2019 observation-interface protocol.

The candidate manifest is the single source of truth. This launcher resolves a
protocol into the production runner's observation-space vocabulary and records
the exact resolved configuration before loading the diffusion model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path("/depot/rmaulik/data/yangxu")
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts"))

from independent_year_common import six_hour_index  # noqa: E402
from igra_gen import run_aircraft_13var_persistent as production  # noqa: E402


DEFAULT_MANIFEST = REPO / "configs/independent_year_2019/candidate_manifest_2019_v1.json"
DEFAULT_IGRA = (
    ROOT
    / "processed/observation_interface_2019/igra_13var_2019_strat24_noaa_v22_rebuild_v1/igra_2019_strat24.pkl"
)
AIRCRAFT_ROOTS = {
    "around5": ROOT
    / "processed/observation_interface_2019/aircraft_madis_abo_13var_2019_strat24_around5_all_sources_v1",
    "around25": ROOT
    / "processed/observation_interface_2019/aircraft_madis_abo_13var_2019_strat24_around25_all_sources_v1",
}
DEFAULT_METAR = (
    ROOT
    / "processed/observation_interface_2019/surface_madis_metar_13var_2019_strat24_strict_conus_v1"
)

AIRCRAFT_OPERATORS = {
    "v1_pointwise": ("aircraft_simple_sparse", {}),
    "v2_cell_balanced_pointwise": ("aircraft_weighted_sparse", {}),
    "v4_equal_cell": ("aircraft_superob_grid", {"aircraft_grid_aggregation": "equal"}),
    "v4c_distance_weighted_cell": (
        "aircraft_superob_grid",
        {"aircraft_grid_aggregation": "distance", "aircraft_distance_weight_sigma_cell": 1.0},
    ),
}
METAR_OPERATORS = {
    "v1_pointwise": ("surface_simple_sparse", {}),
    "v2_cell_balanced_pointwise": ("surface_weighted_sparse", {}),
    "v4_equal_cell": ("surface_superob_grid", {"surface_grid_aggregation": "equal"}),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_manifest(path: Path) -> tuple[dict, str]:
    digest = sha256(path)
    sha_path = path.with_suffix(".sha256")
    if not sha_path.exists():
        raise FileNotFoundError(f"Missing frozen-manifest checksum: {sha_path}")
    expected = sha_path.read_text().split()[0]
    if digest != expected:
        raise ValueError(f"Candidate manifest checksum mismatch: expected {expected}, found {digest}")
    return json.loads(path.read_text()), digest


def find_protocol(manifest: dict, protocol_id: str) -> dict:
    matches = [item for item in manifest["protocols"] if item["protocol_id"] == protocol_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one protocol_id={protocol_id!r}; found {len(matches)}")
    return matches[0]


def resolve_structure(protocol: dict, manifest: dict) -> tuple[dict, dict]:
    stage = protocol["stage"]
    if stage == "baseline":
        return dict(production.EXPERIMENTS["igra_only"]), {}

    if stage == "aircraft_structure":
        operator = protocol["aircraft_operator"]
        obs_space, operator_settings = AIRCRAFT_OPERATORS[operator]
        window = protocol["aircraft_pressure_window"]
        source_policy = protocol["aircraft_source_policy"]
        spec = {
            "use_igra": True,
            "obs_mode": f"aircraft_{window}",
            "obs_modality": "aircraft",
            "obs_space": obs_space,
            "aircraft_data_sources": manifest["aircraft_source_policies"][source_policy],
            "aircraft_spatial_support": manifest["fixed_sampling"]["support"],
            "aircraft_pressure_window_name": window,
            "aircraft_source_policy_name": source_policy,
            "description": (
                f"2019 development: IGRA plus aircraft {operator}; source_policy={source_policy}; "
                f"pressure_window={window}."
            ),
            **operator_settings,
        }
        return spec, {"aircraft": protocol["aircraft_parameters"]}

    if stage == "metar_structure":
        operator = protocol["metar_operator"]
        obs_space, operator_settings = METAR_OPERATORS[operator]
        spec = {
            "use_igra": True,
            "obs_mode": "surface_metar_strat24",
            "obs_modality": "surface",
            "obs_space": obs_space,
            "surface_variables": production.SURFACE_METAR_VARIABLES,
            "surface_spatial_support": manifest["fixed_sampling"]["support"],
            "description": f"2019 development: IGRA plus METAR {operator}.",
            **operator_settings,
        }
        return spec, {"surface": protocol["metar_parameters"]}

    raise ValueError(
        f"Stage {stage!r} requires a prior-stage selection artifact and is intentionally not "
        "runnable through the structural-stage launcher."
    )


def six_hour_index_text(timestamp: str) -> int:
    from datetime import datetime

    return six_hour_index(datetime.fromisoformat(timestamp.removesuffix("Z")))


def validate_inputs(protocol: dict, manifest: dict, igra_pkl: Path, metar_root: Path) -> list[int]:
    timesteps = [six_hour_index_text(text) for text in manifest["development_timestamps"]]
    missing: list[str] = []
    if not igra_pkl.is_file():
        missing.append(str(igra_pkl))
    if protocol["stage"] == "aircraft_structure":
        root = AIRCRAFT_ROOTS[protocol["aircraft_pressure_window"]]
        for timestep in timesteps:
            path = root / "obs" / f"madis_aircraft_13var_t{timestep:04d}.npz"
            if not path.is_file():
                missing.append(str(path))
    if protocol["stage"] == "metar_structure":
        for timestep in timesteps:
            path = metar_root / "obs" / f"madis_metar_surface_13var_t{timestep:04d}.npz"
            if not path.is_file():
                missing.append(str(path))
    era5_val = Path(production.ERA5_ROOT) / "val"
    for timestep in timesteps:
        if not (era5_val / f"2019_{timestep:04d}.h5").is_file():
            missing.append(str(era5_val / f"2019_{timestep:04d}.h5"))
    if missing:
        raise FileNotFoundError("Missing required 2019 inputs:\n" + "\n".join(missing))
    return timesteps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--igra-pkl", type=Path, default=DEFAULT_IGRA)
    parser.add_argument("--metar-root", type=Path, default=DEFAULT_METAR)
    parser.add_argument("--timesteps", default="manifest")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run t0000 with ensemble=1 and steps=5; never use smoke outputs for ranking.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest, manifest_sha256 = load_frozen_manifest(args.candidate_manifest)
    protocol = find_protocol(manifest, args.protocol_id)
    spec, modality_params = resolve_structure(protocol, manifest)
    manifest_timesteps = validate_inputs(protocol, manifest, args.igra_pkl, args.metar_root)
    if args.smoke:
        timesteps = [manifest_timesteps[0]]
    elif args.timesteps == "manifest":
        timesteps = manifest_timesteps
    else:
        timesteps = production.parse_timesteps(args.timesteps)
        unexpected = sorted(set(timesteps) - set(manifest_timesteps))
        if unexpected:
            raise ValueError(f"Requested timesteps outside frozen 2019 development set: {unexpected}")

    protocol_root = args.run_root / "protocols" / args.protocol_id
    effective_sampling = dict(manifest["fixed_sampling"])
    if args.smoke:
        effective_sampling.update({"ensemble": 1, "steps": 5, "mode": "code_path_smoke_not_for_ranking"})
    else:
        effective_sampling["mode"] = "frozen_development_protocol"
    resolved = {
        "candidate_manifest": str(args.candidate_manifest.resolve()),
        "candidate_manifest_sha256": manifest_sha256,
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
        "effective_sampling": effective_sampling,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return

    protocol_root.mkdir(parents=True, exist_ok=True)
    (protocol_root / "resolved_protocol.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    production.EXPERIMENTS[args.protocol_id] = spec

    fixed_igra = manifest["fixed_igra"]
    sampling = effective_sampling
    aircraft = modality_params.get("aircraft", {"lambda": 0.1, "std": 5e-4, "gamma": 5e-6})
    surface = modality_params.get("surface", {"lambda": 0.2, "std": 5e-4, "gamma": 1e-5})
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
