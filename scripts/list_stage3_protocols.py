#!/usr/bin/env python3
"""List the 54 frozen Stage III numerical-calibration protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_independent_year_2019_protocol import DEFAULT_MANIFEST, load_frozen_manifest


STAGES = {"aircraft_numerical_calibration", "metar_numerical_calibration"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=int)
    parser.add_argument(
        "--representative-centers",
        action="store_true",
        help="Return only the A1 and M5 center protocols for code-path smoke tests.",
    )
    args = parser.parse_args()

    manifest, digest = load_frozen_manifest(args.manifest)
    protocols = [item for item in manifest["protocols"] if item["stage"] in STAGES]
    if args.representative_centers:
        wanted = {
            "aircraft_cal__lambda0.1__std0.0005__gamma5e-06",
            "metar_cal__lambda0.2__std0.0005__gamma1e-05",
        }
        protocols = [item for item in protocols if item["protocol_id"] in wanted]
    if args.index is not None:
        print(protocols[args.index]["protocol_id"])
        return
    print(
        json.dumps(
            {
                "manifest_sha256": digest,
                "count": len(protocols),
                "protocols": protocols,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
