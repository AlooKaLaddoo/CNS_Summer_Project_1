#!/bin/bash

# Minimal batch script to run Granger causality analysis on all EDF files
cd "$(dirname "$0")"

# Process all EDF files in the dataset
find Dataset/Infants_data -name "*.edf" -type f | while read -r edf_file; do
    echo "Processing: $edf_file"
    python granger_causality_analysis.py "$edf_file" || echo "Failed: $edf_file"
done

echo "Batch processing complete!"
