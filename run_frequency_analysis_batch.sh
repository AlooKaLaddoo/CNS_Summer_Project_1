#!/bin/bash

# Batch processing script for EEG frequency band analysis
# Usage: ./run_frequency_analysis_batch.sh

echo "Starting batch processing of EEG frequency band analysis..."

# Find all EDF files in the dataset
EDF_FILES=$(find ./Dataset/Infants_data -name "*.edf" -type f)

# Count total files
TOTAL_FILES=$(echo "$EDF_FILES" | wc -l)
echo "Found $TOTAL_FILES EDF files to process"

# Process each file
CURRENT=1
for edf_file in $EDF_FILES; do
    echo "[$CURRENT/$TOTAL_FILES] Processing: $edf_file"
    
    # Run the analysis
    python individual_frequency_bands_analysis.py "$edf_file"
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed: $edf_file"
    else
        echo "✗ Error processing: $edf_file"
    fi
    
    ((CURRENT++))
    echo "----------------------------------------"
done

echo "Batch processing completed!"
echo "Results are saved in ./frequency_band_analysis_output/"
