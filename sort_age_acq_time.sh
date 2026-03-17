#!/usr/bin/env bash
set -euo pipefail

# Build a global, age-sorted table across all session scan files.
ROOT_DIR="${1:-Infants_data}"
OUTPUT_FILE="${2:-Dataset_info/age_acq_time_sorted.tsv}"

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Error: root directory not found: $ROOT_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

tmp_data="$(mktemp)"
trap 'rm -f "$tmp_data"' EXIT

# Columns in intermediate data:
# subject_id, session_id, filename, age_acq_time, scans_tsv_path
while IFS= read -r scan_file; do
  subject_id="$(basename "$(dirname "$(dirname "$scan_file")")")"
  session_id="$(basename "$(dirname "$scan_file")")"

  awk -F '\t' -v OFS='\t' -v subject="$subject_id" -v session="$session_id" -v scan_path="$scan_file" '
    NR > 1 && NF > 0 {
      age = $2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", age)
      # Keep only numeric ages; skip empty or malformed rows.
      if (age ~ /^[0-9]*\.?[0-9]+$/) {
        print subject, session, $1, age, scan_path
      }
    }
  ' "$scan_file" >> "$tmp_data"
done < <(find "$ROOT_DIR" -type f -name '*_scans.tsv' | sort)

{
  printf 'temporal_order\tsubject_id\tsession_id\tfilename\tage_acq_time\tscans_tsv_path\n'
  sort -t $'\t' -k4,4g -k1,1 -k2,2 "$tmp_data" | awk -F '\t' -v OFS='\t' '{print NR, $1, $2, $3, $4, $5}'
} > "$OUTPUT_FILE"

echo "Wrote sorted table: $OUTPUT_FILE"
wc -l "$OUTPUT_FILE"
