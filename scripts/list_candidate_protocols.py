#!/usr/bin/env python3
"""List or index protocols from the checksummed 2019 candidate manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_independent_year_2019_protocol import DEFAULT_MANIFEST, load_frozen_manifest


GROUPS = {
    "structural": {"baseline", "aircraft_structure", "metar_structure"},
    "structural_smoke": {"aircraft_structure", "metar_structure"},
    "aircraft_numerical": {"aircraft_numerical_calibration"},
    "metar_numerical": {"metar_numerical_calibration"},
    "joint": {"joint_interaction"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--representative-smokes", action="store_true")
    args = parser.parse_args()

    manifest, digest = load_frozen_manifest(args.manifest)
    protocols = [item for item in manifest["protocols"] if item["stage"] in GROUPS[args.group]]
    if args.representative_smokes:
        wanted = {
            "aircraft_struct__all_qc__around25__v1_pointwise",
            "aircraft_struct__all_qc__around25__v2_cell_balanced_pointwise",
            "aircraft_struct__all_qc__around25__v4_equal_cell",
            "aircraft_struct__all_qc__around25__v4c_distance_weighted_cell",
            "metar_struct__v1_pointwise",
            "metar_struct__v2_cell_balanced_pointwise",
            "metar_struct__v4_equal_cell",
        }
        protocols = [item for item in protocols if item["protocol_id"] in wanted]
    if args.index is not None:
        print(protocols[args.index]["protocol_id"])
        return
    print(json.dumps({"manifest_sha256": digest, "count": len(protocols), "protocols": protocols}, indent=2))


if __name__ == "__main__":
    main()
