#!/usr/bin/env python3
"""CPU smoke tests for 2019 date mapping, source filtering, and H interfaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/depot/rmaulik/data/yangxu")
REPO = ROOT / "repos/observation_interface_independent_year_2019"
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from igra_gen import run_aircraft_13var_persistent as production  # noqa: E402
from run_independent_year_2019_protocol import (  # noqa: E402
    DEFAULT_MANIFEST,
    find_protocol,
    load_frozen_manifest,
    resolve_structure,
)


def lightweight_runner():
    runner = production.PersistentFullPoolRunner.__new__(production.PersistentFullPoolRunner)
    runner.calendar_year = 2019
    runner.model_vars = production.IGRA_VARIABLES
    runner.aircraft_roots = {
        "aircraft_around5": str(
            ROOT
            / "processed/observation_interface_2019/aircraft_madis_abo_13var_2019_strat24_around5_all_sources_v1"
        ),
        "aircraft_around25": str(
            ROOT
            / "processed/observation_interface_2019/aircraft_madis_abo_13var_2019_strat24_around25_all_sources_v1"
        ),
    }
    runner.surface_metar_root = str(
        ROOT
        / "processed/observation_interface_2019/surface_madis_metar_13var_2019_strat24_strict_conus_v1"
    )
    runner.channel_mean = {name: 0.0 for name in runner.model_vars}
    runner.channel_std = {name: 1.0 for name in runner.model_vars}
    era5 = ROOT / "data_from_DJ_original_NERSC/1.40625deg_from_full_res_1_step_6hr_h5df"
    runner.era5_lat = np.load(era5 / "lat.npy").astype(np.float32)
    runner.era5_lon = np.load(era5 / "lon.npy").astype(np.float32)
    return runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest, digest = load_frozen_manifest(DEFAULT_MANIFEST)
    runner = lightweight_runner()
    assert runner.timestep_to_datetime(56).isoformat() == "2019-01-15T00:00:00"

    representative = [
        "baseline_igra_only",
        "aircraft_struct__all_qc__around5__v1_pointwise",
        "aircraft_struct__exclude_tamdar__around25__v2_cell_balanced_pointwise",
        "aircraft_struct__legacy_keep015__around25__v4_equal_cell",
        "aircraft_struct__all_qc__around25__v4c_distance_weighted_cell",
        "metar_struct__v1_pointwise",
        "metar_struct__v2_cell_balanced_pointwise",
        "metar_struct__v4_equal_cell",
    ]
    resolved = {}
    for protocol_id in representative:
        spec, parameters = resolve_structure(find_protocol(manifest, protocol_id), manifest)
        resolved[protocol_id] = {"obs_space": spec.get("obs_space", "points"), "parameters": parameters}

    source_results = {}
    policies = {
        "all_qc": [0, 1, 3, 4, 5, 6],
        "exclude_tamdar": [0, 1, 3, 5, 6],
        "legacy_keep015": [0, 1, 5],
    }
    for name, codes in policies.items():
        metadata, _, _, _, _ = runner.load_aircraft_measurements(
            "aircraft_around25",
            0,
            spatial_support="strict_conus",
            weighting="cell_balanced",
            return_pressure=True,
            data_sources=codes,
        )
        source_counts = json.loads(metadata["aircraft_source_counts_json"])
        observed = sorted(
            {
                int(key.split("_")[1])
                for channel in source_counts.values()
                for key in channel
                if key.startswith("dataSource_")
            }
        )
        assert set(observed) <= set(codes)
        source_results[name] = observed

    _, _, aircraft_masks, _, _ = runner.load_aircraft_superob_grid(
        "aircraft_around25",
        0,
        spatial_support="strict_conus",
        aggregation="equal",
        data_sources=policies["legacy_keep015"],
    )
    metar_metadata, _, _, _ = runner.load_surface_metar_measurements(
        0,
        variables=production.SURFACE_METAR_VARIABLES,
        weighting="cell_balanced",
        spatial_support="strict_conus",
    )
    result = {
        "status": "PASS",
        "candidate_manifest_sha256": digest,
        "resolved_representative_protocols": resolved,
        "observed_sources_after_filtering": source_results,
        "aircraft_v4_valid_cells_t0000": sum(int(mask.sum()) for mask in aircraft_masks),
        "metar_points_t0000": sum(json.loads(metar_metadata["surface_counts_json"]).values()),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
