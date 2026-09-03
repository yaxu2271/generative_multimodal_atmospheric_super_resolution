#!/usr/bin/env python3
"""Inspect small MADIS aircraft netCDF(.gz) samples.

This script is intentionally lightweight. It reads compressed MADIS aircraft
files, reports dimensions/variables, and summarizes a small set of variables
needed for 13-var feasibility.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


KEY_CANDIDATES = [
    "T",
    "temperature",
    "U",
    "u",
    "V",
    "v",
    "DD",
    "windDir",
    "FF",
    "windSpeed",
    "Q",
    "RH",
    "relHumidity",
    "TD",
    "dewpoint",
    "P",
    "pressure",
    "HT",
    "altitude",
    "GPSHT",
    "GPSaltitude",
    "LAT",
    "latitude",
    "LON",
    "longitude",
    "TDAYSEC",
    "timeObs",
    "observationTime",
]


def _decode_to_temp(path: Path) -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="madis_aircraft_", suffix=".nc", delete=False)
    tmp_path = Path(tmp.name)
    with gzip.open(path, "rb") as src:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
    tmp.close()
    return tmp_path


def _to_float_array(var: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(var[:])
    except Exception:
        return None
    if arr.dtype.kind not in "fiu":
        return None
    arr = arr.astype(np.float64, copy=False)
    fill = getattr(var, "_FillValue", None)
    if fill is not None:
        arr = np.where(arr == float(fill), np.nan, arr)
    missing = getattr(var, "missing_value", None)
    if missing is not None:
        try:
            arr = np.where(arr == float(missing), np.nan, arr)
        except Exception:
            pass
    arr = np.where(arr <= -9990.0, np.nan, arr)
    return arr


def _summarize_numeric(var: Any) -> dict[str, Any] | None:
    arr = _to_float_array(var)
    if arr is None:
        return None
    finite = np.isfinite(arr)
    n_finite = int(finite.sum())
    out: dict[str, Any] = {
        "shape": list(arr.shape),
        "units": getattr(var, "units", ""),
        "long_name": getattr(var, "long_name", getattr(var, "standard_name", "")),
        "n_total": int(arr.size),
        "n_finite": n_finite,
    }
    if n_finite:
        vals = arr[finite]
        out.update(
            {
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "mean": float(np.nanmean(vals)),
            }
        )
    return out


def inspect_file(path: Path) -> dict[str, Any]:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise SystemExit("netCDF4 is required in the active Python environment") from exc

    tmp_path = _decode_to_temp(path)
    try:
        with Dataset(tmp_path, "r") as ds:
            dims = {name: len(dim) for name, dim in ds.dimensions.items()}
            variables = sorted(ds.variables.keys())
            key_summaries = {}
            for key in KEY_CANDIDATES:
                if key in ds.variables:
                    summary = _summarize_numeric(ds.variables[key])
                    if summary is not None:
                        key_summaries[key] = summary
            return {
                "path": str(path),
                "compressed_size_bytes": path.stat().st_size,
                "dimensions": dims,
                "n_variables": len(variables),
                "variables": variables,
                "key_summaries": key_summaries,
            }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def write_key_csv(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "variable",
                "shape",
                "units",
                "n_total",
                "n_finite",
                "min",
                "max",
                "mean",
            ],
        )
        writer.writeheader()
        for result in results:
            for name, summary in result["key_summaries"].items():
                writer.writerow(
                    {
                        "file": result["path"],
                        "variable": name,
                        "shape": "x".join(map(str, summary.get("shape", []))),
                        "units": summary.get("units", ""),
                        "n_total": summary.get("n_total", ""),
                        "n_finite": summary.get("n_finite", ""),
                        "min": _fmt(summary.get("min")),
                        "max": _fmt(summary.get("max")),
                        "mean": _fmt(summary.get("mean")),
                    }
                )


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{value:.8g}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    results = [inspect_file(path) for path in args.files]
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True))
    write_key_csv(results, args.csv_out)

    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.csv_out}")
    for result in results:
        print(
            f"{Path(result['path']).name}: "
            f"dims={result['dimensions']} n_variables={result['n_variables']}"
        )
        print("  key vars:", ", ".join(result["key_summaries"].keys()) or "none")


if __name__ == "__main__":
    main()
