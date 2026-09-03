#!/usr/bin/env python3
"""List frozen Stage III-D directional-boundary protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_independent_year_2019_protocol import sha256


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO
    / "configs/independent_year_2019/stage3d_directional_boundary_manifest_v1.json"
)


def load_manifest(path: Path) -> tuple[dict, str]:
    digest = sha256(path)
    checksum = path.with_suffix(".sha256")
    if not checksum.is_file():
        raise FileNotFoundError(f"Missing Stage III-D checksum: {checksum}")
    expected = checksum.read_text().split()[0]
    if digest != expected:
        raise ValueError(
            f"Stage III-D checksum mismatch: expected {expected}, found {digest}"
        )
    manifest = json.loads(path.read_text())
    if manifest["new_protocol_count"] != len(manifest["protocols"]):
        raise ValueError("Stage III-D manifest count mismatch")
    if manifest["new_protocol_count"] != 34:
        raise ValueError("Stage III-D manifest must contain exactly 34 protocols")
    return manifest, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()

    manifest, digest = load_manifest(args.manifest)
    protocols = manifest["protocols"]
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
