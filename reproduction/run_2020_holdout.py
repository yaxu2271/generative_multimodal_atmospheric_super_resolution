#!/usr/bin/env python3
"""Run the paper's 24-case aircraft or surface-station holdout configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


CONFIG_PATH = REPO_ROOT / "reproduction/config/selected_interface_2019.json"
TIMESTEP_PATH = REPO_ROOT / "reproduction/manifests/holdout_timesteps_2020.json"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", choices=["aircraft", "surface"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hydra-config", type=Path, required=True)
    parser.add_argument("--era5-root", type=Path, required=True)
    parser.add_argument("--igra-pkl", type=Path, required=True)
    parser.add_argument(
        "--aircraft-root",
        type=Path,
        required=True,
        help="Full aircraft root, or the retained-80%% root for aircraft holdout.",
    )
    parser.add_argument(
        "--surface-root",
        type=Path,
        required=True,
        help="Full surface root, or the retained-80%% root for surface holdout.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--interface-config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--timesteps", type=Path, default=TIMESTEP_PATH)
    parser.add_argument("--ensemble", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def register_experiment(holdout: str, production) -> str:
    name = f"paper_2020_{holdout}_holdout"
    production.EXPERIMENTS[name] = {
        "use_igra": True,
        "obs_mode": "aircraft_around25",
        "obs_modality": "aircraft_surface",
        "obs_space": "aircraft_surface_superob_grid",
        "aircraft_source_filter": "combined",
        "aircraft_spatial_support": "strict_conus",
        "aircraft_grid_aggregation": "equal",
        "aircraft_pressure_window_name": "around25",
        "aircraft_source_policy_name": "legacy_keep015",
        "surface_variables": production.SURFACE_METAR_VARIABLES,
        "surface_spatial_support": "strict_conus",
        "surface_grid_aggregation": "equal",
        "description": f"Paper 2020 {holdout} retained-80% conditioning run.",
    }
    return name


def main() -> None:
    args = parse_args()
    from igra_gen import run_aircraft_13var_persistent as production

    config = load_json(args.interface_config)
    selected = config["selected_parameters"]
    timesteps = [int(value) for value in load_json(args.timesteps)["timesteps"]]
    experiment = register_experiment(args.holdout, production)
    output_root = args.output_root / experiment
    output_root.mkdir(parents=True, exist_ok=True)
    resolved = {
        "holdout": args.holdout,
        "experiment": experiment,
        "timesteps": timesteps,
        "ensemble": args.ensemble,
        "steps": args.steps,
        "seed": args.seed,
        "source_commit": config["source_commit"],
        "note": (
            "For the selected source, --aircraft-root or --surface-root must "
            "contain the retained 80% conditioning targets. Excluded targets "
            "are evaluated separately from the generated reconstructions."
        ),
    }
    (output_root / "resolved_run.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )

    runner = production.PersistentFullPoolRunner(
        output_root=str(output_root),
        ens=args.ensemble,
        seed=args.seed,
        num_steps=args.steps,
        igra_pkl=str(args.igra_pkl),
        aircraft_around25_root=str(args.aircraft_root),
        surface_metar_root=str(args.surface_root),
        checkpoint=str(args.checkpoint),
        era5_root=str(args.era5_root),
        hydra_cfg=str(args.hydra_config),
        likelihood_mode="multimodal",
        std_igra=float(selected["igra_std"]),
        gamma_igra=float(selected["igra_gamma"]),
        lambda_igra=float(selected["igra_lambda"]),
        std_aircraft=float(selected["aircraft_std"]),
        gamma_aircraft=float(selected["aircraft_gamma"]),
        lambda_aircraft=float(selected["aircraft_lambda"]),
        std_surface=float(selected["surface_std"]),
        gamma_surface=float(selected["surface_gamma"]),
        lambda_surface=float(selected["surface_lambda"]),
        era5_split="test",
        calendar_year=2020,
    )
    runner.run_many([experiment], timesteps, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
