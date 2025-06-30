# EEG Frequency Band Analysis with Granger Causality

This script performs comprehensive EEG analysis by decomposing signals into individual frequency bands and applying Granger causality analysis to detect directed connectivity patterns.

## Features

1. **Frequency Band Decomposition**: Separates EEG signals into:
   - Delta (0.5-4 Hz)
   - Theta (4-8 Hz) 
   - Alpha (8-13 Hz)
   - Beta (13-30 Hz)
   - Gamma (30-45 Hz)

2. **Time Window Analysis**: Creates overlapping time windows (10s with 30% overlap by default)

3. **Granger Causality**: Tests for directed connectivity between all channel pairs

4. **Network Analysis**: Identifies top significant connections and creates network graphs

5. **Visualizations**:
   - Network graphs for each frequency band
   - Combined connectivity heatmaps
   - Brain region connectivity analysis

6. **Parallel Processing**: Utilizes all CPU cores for efficient computation

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Single File Analysis
```bash
python individual_frequency_bands_analysis.py <path_to_edf_file>
```

Example:
```bash
python individual_frequency_bands_analysis.py "./Dataset/Infants_data/sub-NORB00001/ses-1/eeg/sub-NORB00001_ses-1_task-EEG_eeg.edf"
```

### Batch Processing
```bash
./run_frequency_analysis_batch.sh
```

## Output

For each subject, the following files are generated in `./frequency_band_analysis_output/subject_id/`:

### Visualizations
- `{band}_network.png`: Network graph for each frequency band
- `connectivity_heatmap_all_bands.png`: Combined heatmap across all bands

### Data Files
- `{band}_connections.csv`: Connections for each frequency band
- `all_connections.csv`: Combined connections across all bands
- `analysis_log.txt`: Detailed analysis log and statistics

### CSV Format
Each CSV contains:
- `source`: Source electrode
- `target`: Target electrode  
- `frequency`: Number of times connection was significant
- `band`: Frequency band name

## Configuration

Modify parameters in the `CONFIG` dictionary at the top of the script:

```python
CONFIG = {
    # EEG Processing
    'sampling_freq': 200,           # Target sampling frequency (Hz)
    
    # Frequency Bands (Hz) - customize as needed
    'frequency_bands': {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    },
    
    # Time Window Analysis
    'window_length': 10,            # Time window length (seconds)
    'overlap_ratio': 0.3,           # Window overlap ratio (0-1)
    
    # Granger Causality
    'max_lag': 5,                   # Maximum lag for Granger causality
    'alpha': 0.05,                  # Significance threshold
    
    # Network Analysis
    'top_connections': 10,          # Top N connections to visualize
    
    # Processing
    'n_cores': mp.cpu_count(),      # Number of CPU cores to use
}
```

## Analysis Pipeline

1. **Data Loading**: Loads EEG data from EDF files and extracts EEG channels
2. **Preprocessing**: Filters and resamples data to target frequency
3. **Band Filtering**: Applies bandpass filters for each frequency band
4. **Window Creation**: Creates overlapping time windows
5. **Granger Causality**: Tests all channel pairs for each window
6. **Statistical Analysis**: Identifies significant connections
7. **Visualization**: Creates network graphs and heatmaps
8. **Export**: Saves results to CSV files and generates log

## Performance

- **Multiprocessing**: Uses all available CPU cores
- **Memory Optimization**: Processes data in chunks to handle large files
- **Parallel GC Computation**: Granger causality tests run in parallel

## Expected Runtime

For a typical 10-minute EEG recording with 21 channels:
- Single subject: ~20-30 minutes (depends on CPU cores)
- Memory usage: ~2-4 GB peak

## Notes

- Script automatically handles BIDS-formatted filenames
- Output directories are created automatically
- All results include detailed logging for reproducibility
- Network graphs show only top connections to avoid cluttered visualization
- CSV files contain complete connectivity information for further analysis
