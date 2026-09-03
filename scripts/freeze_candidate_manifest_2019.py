#!/usr/bin/env python3
"""Write the prespecified 2019 observation-interface candidate manifest."""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

from independent_year_common import stratified24


ROOT = Path("/depot/rmaulik/data/yangxu")
REPO = ROOT / "repos/observation_interface_independent_year_2019"
OUT = REPO / "configs/independent_year_2019/candidate_manifest_2019_v1.json"

SOURCE_POLICIES = {
    "all_qc": [0, 1, 3, 4, 5, 6],
    "exclude_tamdar": [0, 1, 3, 5, 6],
    "legacy_keep015": [0, 1, 5],
}
WINDOWS = {
    "around5": {"500": [495.0, 505.0], "850": [845.0, 855.0]},
    "around25": {"500": [475.0, 525.0], "850": [825.0, 875.0]},
}
OPERATORS = ["v1_pointwise", "v2_cell_balanced_pointwise", "v4_equal_cell", "v4c_distance_weighted_cell"]
METAR_OPERATORS = ["v1_pointwise", "v2_cell_balanced_pointwise", "v4_equal_cell"]


def protocol(protocol_id: str, stage: str, **settings):
    return {"protocol_id": protocol_id, "stage": stage, **settings}


def main() -> None:
    protocols = []
    aircraft_center = {"lambda": 0.10, "std": 5e-4, "gamma": 5e-6}
    metar_center = {"lambda": 0.20, "std": 5e-4, "gamma": 1e-5}

    protocols.append(protocol("baseline_igra_only", "baseline", modalities=["igra"]))
    for source, window, operator in itertools.product(SOURCE_POLICIES, WINDOWS, OPERATORS):
        protocols.append(
            protocol(
                f"aircraft_struct__{source}__{window}__{operator}",
                "aircraft_structure",
                modalities=["igra", "aircraft"],
                aircraft_source_policy=source,
                aircraft_pressure_window=window,
                aircraft_operator=operator,
                aircraft_parameters=aircraft_center,
            )
        )
    for operator in METAR_OPERATORS:
        protocols.append(
            protocol(
                f"metar_struct__{operator}",
                "metar_structure",
                modalities=["igra", "metar"],
                metar_operator=operator,
                metar_parameters=metar_center,
            )
        )

    for lam, std, gamma in itertools.product([0.05, 0.10, 0.20], [2.5e-4, 5e-4, 1e-3], [2e-6, 5e-6, 1e-5]):
        protocols.append(
            protocol(
                f"aircraft_cal__lambda{lam:g}__std{std:g}__gamma{gamma:g}",
                "aircraft_numerical_calibration",
                modalities=["igra", "aircraft"],
                structural_settings="selected_by_aircraft_structure_stage",
                aircraft_parameters={"lambda": lam, "std": std, "gamma": gamma},
            )
        )
    for lam, std, gamma in itertools.product([0.10, 0.20, 0.40], [2.5e-4, 5e-4, 1e-3], [5e-6, 1e-5, 2e-5]):
        protocols.append(
            protocol(
                f"metar_cal__lambda{lam:g}__std{std:g}__gamma{gamma:g}",
                "metar_numerical_calibration",
                modalities=["igra", "metar"],
                structural_settings="selected_by_metar_structure_stage",
                metar_parameters={"lambda": lam, "std": std, "gamma": gamma},
            )
        )
    for aircraft_scale, metar_scale in itertools.product([0.5, 1.0, 2.0], repeat=2):
        protocols.append(
            protocol(
                f"interaction__aircraft{aircraft_scale:g}x__metar{metar_scale:g}x",
                "joint_interaction",
                modalities=["igra", "aircraft", "metar"],
                base_settings="separately_selected_2019_aircraft_and_metar_interfaces",
                aircraft_lambda_multiplier=aircraft_scale,
                metar_lambda_multiplier=metar_scale,
            )
        )

    timestamps = [dt.isoformat() + "Z" for dt in stratified24(2019)]
    manifest = {
        "manifest_version": "candidate_manifest_2019_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_split": {"development": 2019, "primary_evaluation": 2020, "future_transfer_not_started": 2025},
        "development_timestamps": timestamps,
        "fixed_sampling": {"ensemble": 16, "steps": 50, "seed": 17, "support": "strict_conus"},
        "fixed_igra": {"lambda": 1.0, "std": 5e-4, "gamma": 2e-6, "operator": "predecessor_pointwise_bilinear"},
        "aircraft_source_policies": SOURCE_POLICIES,
        "source_policy_rationale": {
            "all_qc": "All source codes present in the complete 2019 point/acars audit after product QCR and physical checks.",
            "exclude_tamdar": "Documentation-motivated broad policy excluding 2019 TAMDAR/AirDat code 4 while retaining other audited operational sources.",
            "legacy_keep015": "Disclosed legacy candidate inherited from the exploratory 2020 study; it receives no ranking privilege.",
        },
        "aircraft_pressure_windows": WINDOWS,
        "candidate_counts": {
            "baseline": 1,
            "aircraft_structure": 24,
            "metar_structure": 3,
            "aircraft_numerical_calibration": 27,
            "metar_numerical_calibration": 27,
            "joint_interaction": 9,
            "total": len(protocols),
        },
        "selection_rule": {
            "primary": "mean over 24 timestamps and 13 variables of 100*(RMSE_candidate-RMSE_IGRA)/RMSE_IGRA",
            "paired_uncertainty": "10000 paired bootstrap replicates over 12 month blocks, each block containing the prespecified 1st and 15th; seed=1701",
            "statistical_tie": "95% paired-bootstrap interval for the candidate difference contains zero",
            "secondary": {"aircraft": "constrained6", "metar": "surface3", "joint": "report constrained6 and surface3; use all13 for rank"},
            "source_policy_tie_order": ["all_qc", "exclude_tamdar", "legacy_keep015"],
            "pressure_window_tie_order": ["around5", "around25"],
            "operator_tie_order": ["v4_equal_cell", "v2_cell_balanced_pointwise", "v1_pointwise", "v4c_distance_weighted_cell"],
            "numerical_tie": "minimum log-distance to prespecified center, then weaker guidance in order lower gamma, lower lambda, larger std",
            "interaction_tie": "minimum log-distance to (1x,1x), then lower total lambda multiplier",
            "forbidden": "2020 outcomes may not select candidates or break ties",
        },
        "protocols": protocols,
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(encoded)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    (OUT.parent / "candidate_manifest_2019_v1.sha256").write_text(f"{digest}  {OUT.name}\n")
    print(json.dumps({"output": str(OUT), "sha256": digest, "candidate_counts": manifest["candidate_counts"]}, indent=2))


if __name__ == "__main__":
    main()
