#!/usr/bin/env python3
"""Validate expected Stage IV posterior samples and measurement summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from list_stage4_protocols import DEFAULT_MANIFEST as DEFAULT_STAGE4_MANIFEST, load_manifest
from run_independent_year_2019_protocol import DEFAULT_MANIFEST, load_frozen_manifest, six_hour_index_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage4-manifest", type=Path, default=DEFAULT_STAGE4_MANIFEST)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base, base_digest = load_frozen_manifest(args.base_manifest)
    stage4, stage4_digest = load_manifest(args.stage4_manifest)
    matches = [item for item in stage4["protocols"] if item["protocol_id"] == args.protocol_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one Stage IV protocol for {args.protocol_id!r}")

    timesteps = [six_hour_index_text(text) for text in base["development_timestamps"]]
    ens = int(base["fixed_sampling"]["ensemble"])
    steps = int(base["fixed_sampling"]["steps"])
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
        finite = bool(np.isfinite(array).all())
        if array.shape != (ens, 13, 128, 256) or not finite:
            invalid.append({"path": str(sample), "shape": list(array.shape), "all_finite": finite})
    result = {
        "status": "PASS" if not missing and not invalid else "FAIL",
        "base_candidate_manifest_sha256": base_digest,
        "stage4_manifest_sha256": stage4_digest,
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
