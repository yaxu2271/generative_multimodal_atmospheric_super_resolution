#!/usr/bin/env python3
"""Run the paper's 2020 observation-conditioning configurations.

The primary annual configurations are R, R+A, R+S, and R+A+S. Prior, A, S,
and A+S are provided for the fixed-parameter 24-case removal sensitivity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


CONFIG_PATH = REPO_ROOT / "reproduction/config/selected_interface_2019.json"
PRIOR_CONFIG_PATH = REPO_ROOT / "reproduction/config/atmospheric_prior_13var.yaml"
TIMESTEP_PATH = REPO_ROOT / "reproduction/manifests/evaluation_timesteps_2020.json"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require_path(value: Path | None, flag: str, configuration: str) -> Path:
    if value is None:
        raise SystemExit(f"{flag} is required for configuration {configuration}")
    return value


def register_experiment(
    configuration: str, production, aircraft_data_sources: list[int]
) -> str:
    no_r_names = {
        "Prior": "prior_no_observations",
        "A": "aircraft_only",
        "S": "surface_only",
        "A+S": "aircraft_surface",
    }
    name = no_r_names.get(
        configuration, "paper_2020_" + configuration.replace("+", "plus")
    )
    if configuration == "R":
        spec = {
            "use_igra": True,
            "obs_mode": "none",
            "obs_modality": None,
            "description": "Paper reference configuration: IGRA radiosondes only.",
        }
    elif configuration == "Prior":
        spec = {
            "use_igra": False,
            "obs_mode": "none",
            "obs_modality": None,
            "description": "Pretrained prior without observation likelihood factors.",
        }
    elif configuration in {"R+A", "A"}:
        spec = {
            "use_igra": configuration.startswith("R"),
            "obs_mode": "aircraft_pressure_window_25hpa",
            "obs_modality": "aircraft",
            "obs_space": "aircraft_cell_mean_grid",
            "aircraft_source_filter": "acars",
            "aircraft_spatial_support": "strict_conus",
            "aircraft_grid_aggregation": "equal",
            "aircraft_pressure_window_name": "within_25_hpa",
            "aircraft_data_sources": aircraft_data_sources,
            "description": f"Paper {configuration} configuration.",
        }
    elif configuration in {"R+S", "S"}:
        spec = {
            "use_igra": configuration.startswith("R"),
            "obs_mode": "surface_metar",
            "obs_modality": "surface",
            "obs_space": "surface_cell_mean_grid",
            "surface_variables": production.SURFACE_METAR_VARIABLES,
            "surface_spatial_support": "strict_conus",
            "surface_grid_aggregation": "equal",
            "description": f"Paper {configuration} configuration.",
        }
    else:
        spec = {
            "use_igra": configuration.startswith("R"),
            "obs_mode": "aircraft_pressure_window_25hpa",
            "obs_modality": "aircraft_surface",
            "obs_space": "aircraft_surface_cell_mean_grid",
            "aircraft_source_filter": "acars",
            "aircraft_spatial_support": "strict_conus",
            "aircraft_grid_aggregation": "equal",
            "aircraft_pressure_window_name": "within_25_hpa",
            "aircraft_data_sources": aircraft_data_sources,
            "surface_variables": production.SURFACE_METAR_VARIABLES,
            "surface_spatial_support": "strict_conus",
            "surface_grid_aggregation": "equal",
            "description": f"Paper {configuration} configuration.",
        }
    production.EXPERIMENTS[name] = spec
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration",
        choices=["R", "R+A", "R+S", "R+A+S", "Prior", "A", "S", "A+S"],
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hydra-config", type=Path, default=PRIOR_CONFIG_PATH)
    parser.add_argument("--era5-root", type=Path, required=True)
    parser.add_argument("--igra-pkl", type=Path, required=True)
    parser.add_argument("--aircraft-root", type=Path)
    parser.add_argument("--surface-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--interface-config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--timesteps", type=Path, default=TIMESTEP_PATH)
    parser.add_argument("--ensemble", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from igra_gen import run_aircraft_13var_persistent as production

    if "A" in args.configuration:
        require_path(args.aircraft_root, "--aircraft-root", args.configuration)
    if "S" in args.configuration:
        require_path(args.surface_root, "--surface-root", args.configuration)

    config = load_json(args.interface_config)
    selected = config["selected_parameters"]
    aircraft_data_sources = [
        int(value) for value in config["aircraft_interface"]["madis_data_source_codes"]
    ]
    manifest = load_json(args.timesteps)
    timesteps = [int(value) for value in manifest["timesteps"]]
    ensemble = args.ensemble
    steps = args.steps
    if args.smoke:
        timesteps = timesteps[:1]
        ensemble = 1
        steps = min(5, steps)

    experiment = register_experiment(
        args.configuration, production, aircraft_data_sources
    )
    output_root = args.output_root / experiment
    output_root.mkdir(parents=True, exist_ok=True)
    resolved = {
        "configuration": args.configuration,
        "experiment": experiment,
        "timesteps": timesteps,
        "calendar_year": 2020,
        "ensemble": ensemble,
        "steps": steps,
        "seed": args.seed,
        "interface_config": str(args.interface_config.resolve()),
        "timesteps_manifest": str(args.timesteps.resolve()),
        "hydra_config": str(args.hydra_config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "era5_root": str(args.era5_root.resolve()),
        "igra_pkl": str(args.igra_pkl.resolve()),
        "aircraft_root": str(args.aircraft_root.resolve()) if args.aircraft_root else None,
        "surface_root": str(args.surface_root.resolve()) if args.surface_root else None,
        "output_root": str(output_root.resolve()),
        "likelihood_structure": "source_specific",
        "selected_parameters": selected,
        "aircraft_data_sources": aircraft_data_sources,
    }
    (output_root / "resolved_run.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )

    runner = production.PersistentFullPoolRunner(
        output_root=str(output_root),
        ens=ensemble,
        seed=args.seed,
        num_steps=steps,
        igra_pkl=str(args.igra_pkl),
        aircraft_root=str(args.aircraft_root or Path(".")),
        surface_metar_root=str(args.surface_root or Path(".")),
        checkpoint=str(args.checkpoint),
        era5_root=str(args.era5_root),
        hydra_cfg=str(args.hydra_config),
        likelihood_structure="source_specific",
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
