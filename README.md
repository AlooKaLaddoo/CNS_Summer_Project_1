# Granger Causality Analysis for Infant EEG Data

This repository provides tools for performing Granger causality analysis on infant EEG data, revealing causal relationships between different brain regions and their temporal dynamics.

## Overview

### Python Script (Recommended)
The main analysis tool is `granger_causality_analysis.py` - a command-line script that performs comprehensive Granger causality analysis:

```bash
python granger_causality_analysis.py <path_to_edf_file>
```

**Key Features:**
- **Command-line interface** for easy automation and batch processing
- **Configurable parameters** through a CONFIG dictionary
- **Complete analysis pipeline** with all visualizations and statistics
- **Parallel processing** utilizing all CPU cores for faster computation
- **Organized functions** with single-line docstrings for clarity

### Jupyter Notebook (Development/Interactive)
`Granger_Causality_infants.ipynb` provides the same analysis in an interactive notebook format for exploration and development.

## Analysis Pipeline

Both tools perform the following analyses:

1. **Data Loading & Preprocessing**: Load EDF files, filter signals (0.5-30 Hz), extract EEG channels
2. **Granger Causality Computation**: Calculate pairwise causality with significance testing  
3. **Visualization**: Generate connectivity matrices, network graphs, and brain region analyses
4. **Statistical Analysis**: Provide comprehensive statistics and identify stable connections
5. **Time Window Analysis**: Examine connectivity patterns across time windows
6. **Results Export**: Save all results, plots, and statistical summaries

## Quick Start

### Using the Python Script (Recommended)

```bash
# Basic usage
python granger_causality_analysis.py ./Dataset/Infants_data/sub-NORB00005_ses-1_task-EEG_eeg.edf

# The script will automatically:
# 1. Create an output directory
# 2. Process the EEG data  
# 3. Generate all plots and statistics
# 4. Save results to the output folder
```

### Configuration

Modify parameters in the CONFIG dictionary at the top of `granger_causality_analysis.py`:

```python
CONFIG = {
    'lowpass': 30,          # High-pass filter frequency (Hz)
    'highpass': 0.5,        # Low-pass filter frequency (Hz)  
    'target_fs': 100,       # Target sampling frequency (Hz)
    'max_lag': 2,           # Maximum lag for Granger causality
    'alpha': 0.05,          # Significance threshold
    'window_length': 10,    # Time window length (seconds)
    'overlap_ratio': 0.3,   # Window overlap ratio (0-1)
    'threshold_percentile': 95,  # Network graph threshold percentile
}
```

## Input Data Requirements

- **File type**: EDF (European Data Format)
- **Channel names**: Standard 10-20 EEG electrode names (Fp1, Fp2, F3, F4, etc.)
- **Supported channels**: Fp1, Fp2, F3, F4, F7, F8, FZ, C3, C4, CZ, P3, P4, PZ, O1, O2, T3, T4, T5, T6
- **Minimum duration**: >60 seconds recommended
- **Sampling rate**: >100 Hz recommended

## Parameter Configuration (Python Script)

All parameters can be easily modified in the CONFIG dictionary:

### Processing Parameters
```python
CONFIG = {
    'lowpass': 30,          # Low-pass filter cutoff (Hz)
    'highpass': 0.5,        # High-pass filter cutoff (Hz)
    'target_fs': 100,       # Target sampling frequency (Hz)
    'max_lag': 2,           # Maximum lag for Granger causality
    'alpha': 0.05,          # Statistical significance threshold
}
```

### Time Window Analysis
```python
CONFIG = {
    'window_length': 10,    # Window duration (seconds)
    'overlap_ratio': 0.3,   # Window overlap (0-1)
}
```

### Visualization Settings
```python
CONFIG = {
    'threshold_percentile': 95,  # Network graph threshold
    'max_pairs': 5,             # Max pairs for temporal dynamics
}
```

## Output Files

The analysis creates a subject-specific output directory with:

### Data Files
- `gc_matrix.npy`: Granger causality F-statistic matrix
- `p_values.npy`: Statistical significance p-values  
- `significant_gc.npy`: Significant connections only
- `connections.csv`: Detailed connection information
- `node_stats.csv`: Node connectivity statistics
- `channels.txt`: Channel names used

### Visualization Files
- `gc_matrix_all_connections.png`: Complete connectivity heatmap
- `gc_matrix_significant_connections.png`: Significant connections heatmap
- `connectivity_network.png`: Network graph visualization
- `connectivity_statistics.png`: Statistical plots
- `brain_region_connectivity.png`: Inter-region connectivity
- `connectivity_evolution.png`: Time evolution plots
- `connectivity_stability.png`: Connection consistency analysis

## Usage Examples

### Basic Analysis
```bash
python granger_causality_analysis.py ./Dataset/Infants_data/sub-NORB00005_ses-1_task-EEG_eeg.edf
```

### Batch Processing
```bash
# Process all EDF files in a directory
for file in ./Dataset/Infants_data/*.edf; do
    python granger_causality_analysis.py "$file"
done
```

### Custom Configuration
Edit the CONFIG dictionary in `granger_causality_analysis.py` for different analysis parameters:

```python
# For faster computation (testing)
CONFIG = {
    'target_fs': 50,        # Lower sampling rate
    'max_lag': 1,           # Fewer lags
    'window_length': 5,     # Shorter windows
}

# For detailed analysis  
CONFIG = {
    'max_lag': 5,           # More lags
    'alpha': 0.01,          # Stricter significance
    'window_length': 20,    # Longer windows
}
```

## Dependencies

Install required Python packages:
```bash
pip install numpy pandas matplotlib seaborn mne scipy statsmodels networkx
```

Required packages:
- `numpy`, `pandas`: Data manipulation
- `matplotlib`, `seaborn`: Visualization  
- `mne`: EEG data processing
- `scipy`: Signal processing
- `statsmodels`: Granger causality tests
- `networkx`: Network analysis

## Performance Notes

### Computational Complexity
- Time complexity: O(n² × t × l) where n=channels, t=time points, l=max lag
- Memory usage scales with data length and number of channels
- Parallel processing utilizes all CPU cores for faster computation

### Typical Processing Times
- **Small dataset** (10 channels, 60s): ~2-5 minutes
- **Medium dataset** (15 channels, 300s): ~10-20 minutes  
- **Large dataset** (20+ channels, full recording): ~30+ minutes

### Optimization Tips
- Reduce `target_fs` for faster computation
- Lower `max_lag` for quicker results
- Use shorter `window_length` for initial testing

## Results Interpretation

### Connectivity Matrices
- **Rows**: Source channels (causes) 
- **Columns**: Target channels (effects)
- **Values**: F-statistics (higher = stronger causality)

### Network Graphs
- **Nodes**: EEG channels/brain regions
- **Edges**: Significant causal connections
- **Edge thickness**: Connection strength

### Statistics
- **Out-degree**: How many channels this channel influences
- **In-degree**: How many channels influence this channel  
- **Consistency**: Connection stability across time windows

## Troubleshooting

**Common Issues:**
- **Memory errors**: Reduce `target_fs` or `window_length`
- **No significant connections**: Lower `alpha` threshold or check data quality
- **Channel naming**: Ensure standard 10-20 EEG naming convention
- **File errors**: Verify EDF file format and channel names

## References

1. Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. Econometrica, 37(3), 424-438.
2. Seth, A. K. (2010). A MATLAB toolbox for Granger causal connectivity analysis. Journal of Neuroscience Methods, 186(2), 262-273.
3. Barnett, L., & Seth, A. K. (2014). The MVGC multivariate Granger causality toolbox. Journal of Neuroscience Methods, 223, 50-68.