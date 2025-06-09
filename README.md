# Granger Causality Analysis for Infant EEG Data

This repository contains a comprehensive Jupyter notebook for performing Granger causality analysis on infant EEG data. The analysis reveals causal relationships between different brain regions and their temporal dynamics.

## Overview

The `Granger_Causality_infants.ipynb` notebook performs the following analyses:

1. **Data Loading & Preprocessing**: Loads EEG data from EDF files, filters signals, and classifies channels
2. **Granger Causality Calculation**: Computes pairwise causality between all EEG channels
3. **Visualization**: Creates connectivity matrices, network graphs, and brain region analyses
4. **Statistical Analysis**: Provides comprehensive statistics and identifies stable connections
5. **Time Window Analysis**: Examines how connectivity patterns change across time
6. **Results Export**: Saves all results, plots, and statistical summaries

## Key Features

- **Automated channel classification** (EEG vs. reference vs. artifact channels)
- **Multiple visualization approaches** (heatmaps, network graphs, regional analysis)
- **Time window segmentation** for temporal dynamics analysis
- **Statistical significance testing** with multiple comparison correction
- **Comprehensive output** with all results saved for further analysis

## Input Data Requirements

### Expected File Format
- **File type**: EDF (European Data Format)
- **Naming convention**: `sub-NORB00064_ses-3_task-EEG_eeg.edf`
- **Channel names**: Standard 10-20 EEG electrode names (Fp1, Fp2, F3, F4, etc.)

### Supported Channel Types
- **EEG channels**: Fp1, Fp2, F3, F4, F7, F8, FZ, C3, C4, CZ, P3, P4, PZ, O1, O2, T3, T4, T5, T6, A1, A2
- **Reference channels**: Channels containing 'Pg', 'REF', or 'GND'
- **Artifact channels**: ECG, EOG, EMG channels (automatically excluded)

## Key Parameters You Can Modify

### 1. Data Input Parameters

```python
# Change the input file path
data_path = './Dataset/Infants_data/NORB00064/sub-NORB00064_ses-3_task-EEG_eeg.edf'

# Modify output directory
base_output_dir = './Dataset/Infants_data_output'
```

### 2. Preprocessing Parameters

```python
# Bandpass filter settings
lowpass = 30    # Low-pass filter cutoff (Hz)
highpass = 0.5  # High-pass filter cutoff (Hz)

# Downsampling settings
target_fs = 100  # Target sampling frequency (Hz)

# Data duration for analysis
max_samples = int(320 * new_fs)  # 320 seconds of data
```

### 3. Granger Causality Parameters

```python
# Maximum lag for causality testing
max_lag = 2     # Number of lags to test (affects computation time)

# Statistical significance threshold
alpha = 0.05    # P-value threshold for significance

# Analysis subset (for faster computation)
eeg_subset = eeg_downsampled[:, :max_samples]  # Use subset or full data
```

### 4. Visualization Parameters

```python
# Connectivity matrix visualization
cmap = 'hot'           # Colormap for heatmaps
figsize = (12, 10)     # Figure size

# Network graph parameters
threshold_percentile = 95   # Percentile threshold for network edges
node_size = 1000           # Size of nodes in network graph
edge_alpha = 0.6           # Transparency of edges

# Time window visualization
time_window = slice(0, int(320 * sampling_freq))  # Time range to plot
```

### 5. Time Window Analysis Parameters

```python
# Window segmentation settings
window_length_sec = 60     # Window duration in seconds
overlap_ratio = 0.5        # Overlap between windows (0.5 = 50%)

# Window size comparison
window_sizes = [30, 60, 120]  # Different window sizes to test

# Stability analysis
consistency_threshold = 0.7   # Threshold for connection consistency
```

### 6. Brain Region Analysis

```python
# Define custom brain regions
regions = {
    'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'FZ'],
    'Central': ['C3', 'C4', 'CZ'],
    'Parietal': ['P3', 'P4', 'PZ'],
    'Occipital': ['O1', 'O2'],
    'Temporal': ['T3', 'T4', 'T5', 'T6']
}
```

## Output Files

The analysis automatically creates a subject-specific output directory and saves:

### Data Files
- `gc_matrix.npy`: Full Granger causality F-statistic matrix
- `p_values.npy`: Statistical significance p-values
- `significant_gc.npy`: Thresholded significant connections only
- `channel_names.txt`: List of EEG channel names used
- `significant_connections.csv`: Detailed list of significant connections
- `node_statistics.csv`: Node-wise connectivity statistics

### Time Window Analysis Files
- `gc_windows_60s_matrices.npy`: GC matrices for each time window
- `window_info_60s.csv`: Information about each time window
- `connectivity_metrics_60s.csv`: Connectivity metrics over time
- `stable_connections_across_windows.csv`: Most stable connections
- `connection_consistency_matrix.npy`: Connection consistency across windows

### Visualization Files
- `eeg_sample_data.png`: Sample EEG data traces
- `gc_matrix_all_connections.png`: Complete connectivity matrix heatmap
- `gc_matrix_significant_connections.png`: Significant connections only
- `connectivity_network.png`: Network graph visualization
- `connectivity_statistics.png`: Node degree and connectivity statistics
- `brain_region_connectivity.png`: Inter-region connectivity analysis
- `connectivity_evolution.png`: Connectivity changes over time
- `connectivity_stability.png`: Connection consistency analysis

## Usage Instructions

### 1. Basic Analysis
```python
# Simply change the data_path variable and run all cells
data_path = 'path/to/your/eeg/file.edf'
```

### 2. Faster Computation (for testing)
```python
# Use shorter data duration
max_samples = int(60 * new_fs)  # Use only 60 seconds

# Reduce maximum lag
max_lag = 1

# Use higher downsampling
target_fs = 50  # Lower sampling rate
```

### 3. More Detailed Analysis
```python
# Use full data
eeg_subset = eeg_downsampled  # Use all available data

# Increase maximum lag
max_lag = 5

# Use stricter significance threshold
alpha = 0.01
```

### 4. Custom Time Windows
```python
# Analyze different time scales
window_sizes = [10, 30, 60, 120]  # seconds

# Adjust overlap
overlap_ratio = 0.25  # 25% overlap for less redundancy
```

## Dependencies

The notebook requires the following Python packages:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy import signal
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import os
import re
```

Install dependencies with:
```bash
pip install numpy pandas matplotlib seaborn mne scipy statsmodels networkx
```

## Performance Considerations

### Computational Complexity
- **Time complexity**: O(n² × t × l) where n=channels, t=time points, l=max lag
- **Memory usage**: Scales with data length and number of channels

### Optimization Tips
1. **Downsample data** to reduce computation time
2. **Use shorter time windows** for initial exploration
3. **Reduce max_lag** for faster results
4. **Use data subsets** during development

### Typical Processing Times
- **Small dataset** (10 channels, 60s): ~2-5 minutes
- **Medium dataset** (15 channels, 300s): ~10-20 minutes
- **Large dataset** (20+ channels, full recording): ~30+ minutes

## Interpretation Guide

### Connectivity Matrices
- **Rows**: Source channels (causes)
- **Columns**: Target channels (effects)
- **Values**: F-statistics (higher = stronger causality)

### Network Graphs
- **Nodes**: EEG channels/brain regions
- **Edges**: Significant causal connections
- **Edge thickness**: Connection strength

### Statistical Outputs
- **Out-degree**: How many channels this channel influences
- **In-degree**: How many channels influence this channel
- **Consistency**: How often a connection appears across time windows

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce data length or downsample more aggressively
2. **No significant connections**: Lower alpha threshold or check data quality
3. **Channel naming**: Ensure channels follow standard 10-20 naming convention
4. **File format**: Verify EDF file is properly formatted

### Data Quality Checks
- Ensure adequate sampling rate (>100 Hz recommended)
- Check for excessive noise or artifacts
- Verify channel impedances were acceptable during recording
- Confirm sufficient data length (>60 seconds recommended)

## References

1. Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. Econometrica, 37(3), 424-438.
2. Seth, A. K. (2010). A MATLAB toolbox for Granger causal connectivity analysis. Journal of Neuroscience Methods, 186(2), 262-273.
3. Barnett, L., & Seth, A. K. (2014). The MVGC multivariate Granger causality toolbox. Journal of Neuroscience Methods, 223, 50-68.