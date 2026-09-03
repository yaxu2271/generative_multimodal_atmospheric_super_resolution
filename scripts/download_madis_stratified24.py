#!/usr/bin/env python3
"""Download and validate the prespecified MADIS stratified24 files."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import netCDF4

from independent_year_common import ROOT, stratified24


PRODUCTS = {
    "ABO": "acars",
    "METAR": "metar",
}
REQUIRED = {
    "ABO": {
        "timeObs", "dataSource", "latitude", "longitude", "altitude",
        "temperature", "windSpeed", "windDir", "latitudeQCR",
        "longitudeQCR", "altitudeQCR", "temperatureQCR", "windSpeedQCR",
        "windDirQCR",
    },
    "METAR": {
        "timeObs", "stationName", "latitude", "longitude", "temperature",
        "windSpeed", "windDir", "temperatureQCR", "windSpeedQCR",
        "windDirQCR",
    },
}


def url_for(dt, product: str) -> str:
    return (
        "https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/"
        f"{dt:%Y/%m/%d}/point/{product}/netcdf/{dt:%Y%m%d_%H%M}.gz"
    )


def output_for(root: Path, dt, product: str) -> Path:
    return (
        root
        / "madisPublic1/data/archive"
        / f"{dt:%Y/%m/%d}/point/{product}/netcdf/{dt:%Y%m%d_%H%M}.gz"
    )


def validate_netcdf(compressed: bytes, modality: str) -> tuple[int, list[str]]:
    raw = gzip.decompress(compressed)
    with tempfile.NamedTemporaryFile(suffix=".nc") as handle:
        handle.write(raw)
        handle.flush()
        with netCDF4.Dataset(handle.name) as ds:
            names = set(ds.variables)
            missing = sorted(REQUIRED[modality] - names)
            if missing:
                raise ValueError(f"{modality} file is missing fields: {missing}")
            n_records = len(ds.dimensions["recNum"])
    return n_records, sorted(REQUIRED[modality])


def read_or_download(url: str, path: Path) -> tuple[bytes, str]:
    if path.exists():
        return path.read_bytes(), "existing"
    request = Request(url, headers={"User-Agent": "yangxu-independent-year/1.0"})
    with urlopen(request, timeout=180) as response:
        compressed = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_bytes(compressed)
    os.replace(part, path)
    return compressed, "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data/observation_interface_2019/raw/madis_stratified24_v1",
    )
    args = parser.parse_args()

    rows = []
    for dt in stratified24(args.year):
        for modality, product in PRODUCTS.items():
            url = url_for(dt, product)
            path = output_for(args.raw_root, dt, product)
            compressed, action = read_or_download(url, path)
            n_records, checked_fields = validate_netcdf(compressed, modality)
            rows.append({
                "datetime_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "modality": modality,
                "product": product,
                "url": url,
                "local_path": str(path),
                "action": action,
                "compressed_bytes": len(compressed),
                "sha256": hashlib.sha256(compressed).hexdigest(),
                "n_records": n_records,
                "validated_fields": ";".join(checked_fields),
            })
            print(f"{action:10s} {modality:5s} {dt:%Y-%m-%d %H:%M} n={n_records}")

    manifest_csv = args.raw_root / "download_manifest.csv"
    manifest_json = args.raw_root / "download_manifest.json"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_json.write_text(
        json.dumps({
            "year": args.year,
            "selection_rule": "00 UTC on the 1st and 15th of every month",
            "expected_files": 48,
            "validated_files": len(rows),
            "rows": rows,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_csv}")
    print(f"Wrote {manifest_json}")


if __name__ == "__main__":
    main()
