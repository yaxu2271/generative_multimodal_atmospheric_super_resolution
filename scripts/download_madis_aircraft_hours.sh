#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 OUT_ROOT YYYY-MM-DD[,YYYY-MM-DD...] HH[,HH...]"
  echo "Example: $0 /depot/.../aircraft_madis_abo_raw 2020-01-01,2020-01-02 00,12"
  exit 2
fi

OUT_ROOT="$1"
DATES_CSV="$2"
HOURS_CSV="$3"

BASE_URL="https://madis-data.ncep.noaa.gov/madisPublic1/data/archive"

IFS=',' read -r -a DATES <<< "${DATES_CSV}"
IFS=',' read -r -a HOURS <<< "${HOURS_CSV}"

download_one() {
  local product="$1"
  local date="$2"
  local hour="$3"
  local year month day ymd url out_dir out_file
  year="${date:0:4}"
  month="${date:5:2}"
  day="${date:8:2}"
  ymd="${year}${month}${day}"
  url="${BASE_URL}/${year}/${month}/${day}/point/${product}/netcdf/${ymd}_${hour}00.gz"
  out_dir="${OUT_ROOT}/madisPublic1/archive/${year}/${month}/${day}/point/${product}/netcdf"
  out_file="${out_dir}/${ymd}_${hour}00.gz"
  mkdir -p "${out_dir}"
  if [[ -s "${out_file}" ]]; then
    echo "exists ${out_file}"
    return
  fi
  echo "download ${url}"
  curl --fail --location --retry 3 --retry-delay 2 --max-time 180 --output "${out_file}.tmp" "${url}"
  mv "${out_file}.tmp" "${out_file}"
}

for date in "${DATES[@]}"; do
  for hour in "${HOURS[@]}"; do
    download_one "acars" "${date}" "${hour}"
    download_one "acarsProfiles" "${date}" "${hour}"
  done
done
