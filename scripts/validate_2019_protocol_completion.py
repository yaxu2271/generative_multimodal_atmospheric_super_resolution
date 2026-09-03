#!/usr/bin/env python3
"""Validate expected posterior samples and measurement summaries for one protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from list_stage3d_protocols import (
    DEFAULT_MANIFEST as DEFAULT_STAGE3D_MANIFEST,
    load_manifest as load_stage3d_manifest,
)
from run_independent_year_2019_protocol import (
    DEFAULT_MANIFEST,
    find_protocol,
    load_frozen_manifest,
    six_hour_index_text,
)


def protocol_exists_in_stage3d(stage3d_manifest: Path, protocol_id: str) -> bool:
    stage3d, _ = load_stage3d_manifest(stage3d_manifest)
    return any(item["protocol_id"] == protocol_id for item in stage3d["protocols"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage3d-manifest", type=Path, default=DEFAULT_STAGE3D_MANIFEST)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest, digest = load_frozen_manifest(args.manifest)
    manifest_kind = "base_candidate"
    try:
        find_protocol(manifest, args.protocol_id)
    except ValueError:
        if not protocol_exists_in_stage3d(args.stage3d_manifest, args.protocol_id):
            raise
        manifest_kind = "stage3d_directional_boundary"
    timesteps = [six_hour_index_text(text) for text in manifest["development_timestamps"]]
    ens = int(manifest["fixed_sampling"]["ensemble"])
    steps = int(manifest["fixed_sampling"]["steps"])
    if args.smoke:
        timesteps, ens, steps = timesteps[:1], 1, 5
    sample_dir = args.protocol_root / "samples" / args.protocol_id
    missing = []
    invalid = []
    for timestep in timesteps:
        stem = f"{args.protocol_id}_t{timestep:04d}_e{ens}_s{steps}"
        sample = sample_dir / f"{stem}.npy"
        summary = sample_dir / f"{stem}_measurement_summary.npz"
        if not sample.is_file():
            missing.append(str(sample))
            continue
        if not summary.is_file():
            missing.append(str(summary))
            continue
        array = np.load(sample, mmap_mode="r")
        if array.shape != (ens, 13, 128, 256) or not np.isfinite(array).all():
            invalid.append({"path": str(sample), "shape": list(array.shape), "all_finite": bool(np.isfinite(array).all())})
    result = {
        "status": "PASS" if not missing and not invalid else "FAIL",
        "candidate_manifest_sha256": digest,
        "manifest_kind": manifest_kind,
        "protocol_id": args.protocol_id,
        "expected_timesteps": timesteps,
        "ensemble": ens,
        "steps": steps,
        "missing": missing,
        "invalid": invalid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
