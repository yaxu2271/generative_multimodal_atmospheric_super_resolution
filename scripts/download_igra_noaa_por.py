#!/usr/bin/env python3
"""Download NOAA IGRA period-of-record station files needed for target years."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE_URL = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por"


def station_rows(path: Path, years: set[int]):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if len(line) < 88:
            continue
        try:
            start_year = int(line[72:76])
            end_year = int(line[77:81])
        except ValueError:
            continue
        station_id = line[0:11].strip()
        if not re.fullmatch(r"[A-Z0-9]{11}", station_id):
            continue
        if any(start_year <= year <= end_year for year in years):
            rows.append(
                {
                    "station_id": station_id,
                    "start_year": start_year,
                    "end_year": end_year,
                    "n_soundings_por": int(line[82:88]),
                }
            )
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return len(archive.namelist()) == 1 and archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def download_one(row: dict, output_root: Path, retries: int) -> dict:
    station_id = row["station_id"]
    name = f"{station_id}-data.txt.zip"
    url = f"{BASE_URL}/{name}"
    final = output_root / name
    partial = output_root / f"{name}.part"
    if final.exists() and valid_zip(final):
        return {**row, "url": url, "local_path": str(final), "status": "reused", "bytes": final.stat().st_size, "sha256": sha256(final)}

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url)
            if offset:
                request.add_header("Range", f"bytes={offset}-")
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", 200)
                mode = "ab" if offset and status == 206 else "wb"
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
            if not valid_zip(partial):
                raise zipfile.BadZipFile(f"validation failed for {partial}")
            os.replace(partial, final)
            return {**row, "url": url, "local_path": str(final), "status": "downloaded", "bytes": final.stat().st_size, "sha256": sha256(final)}
        except Exception as exc:  # network failures are recorded and retried
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(30, attempt * 3))
    return {**row, "url": url, "local_path": str(final), "status": "failed", "bytes": 0, "sha256": "", "error": last_error}


def write_manifest(rows: list[dict], csv_path: Path, json_path: Path, years: list[int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": "NOAA NCEI IGRA v2.2 period-of-record sounding files",
        "base_url": BASE_URL,
        "target_years": years,
        "n_requested": len(rows),
        "n_valid": sum(row["status"] in {"reused", "downloaded"} for row in rows),
        "n_failed": sum(row["status"] == "failed" for row in rows),
        "total_bytes": sum(int(row.get("bytes", 0)) for row in rows),
        "manifest_csv": str(csv_path.resolve()),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station_list", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2019, 2020])
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    requested = station_rows(args.station_list, set(args.years))
    print(f"Requesting {len(requested)} station archives with {args.workers} workers", flush=True)
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, row, args.output_root, args.retries) for row in requested]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            completed.append(row)
            if index % 20 == 0 or row["status"] == "failed" or index == len(futures):
                done_gb = sum(int(item.get("bytes", 0)) for item in completed) / 1e9
                failed = sum(item["status"] == "failed" for item in completed)
                print(f"progress={index}/{len(futures)} retained={done_gb:.2f} GB failed={failed}", flush=True)
                write_manifest(sorted(completed, key=lambda item: item["station_id"]), args.manifest_csv, args.summary_json, args.years)


if __name__ == "__main__":
    main()
