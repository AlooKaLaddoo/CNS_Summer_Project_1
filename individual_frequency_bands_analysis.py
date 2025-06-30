#!/usr/bin/env python3
"""
Individual Frequency Bands EEG Analysis with Granger Causality
Usage: python individual_frequency_bands_analysis.py <path_to_edf_file>
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import warnings
from scipy import signal
from scipy.signal import hilbert
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict, Counter
import time

# Configuration - Modify these parameters as needed
CONFIG = {
    # EEG Processing
    'sampling_freq': 200,           # Target sampling frequency (Hz)
    'filter_method': 'fir',         # Filter method ('fir' or 'iir')
    
    # Frequency Bands (Hz)
    'frequency_bands': {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    },
    
    # Time Window Analysis
    'window_length': 4,            # Time window length (seconds)
    'overlap_ratio': 0.3,           # Window overlap ratio (0-1)
    
    # Granger Causality
    'max_lag': 3,                   # Maximum lag for Granger causality
    'alpha': 0.02,                  # Significance threshold
    
    # Network Analysis
    'top_connections': 10,          # Top N connections to visualize
    'network_threshold_percentile': 90,  # Network graph threshold percentile
    
    # Processing
    'n_cores': mp.cpu_count(),      # Number of CPU cores to use
    'use_gpu': False,               # Enable GPU processing (requires CuPy)
    'chunk_size': 100,              # Processing chunk size for memory efficiency
}

def setup_environment():
    """Initialize environment settings."""
    warnings.filterwarnings('ignore')
    mne.set_log_level('WARNING')
    plt.rcParams['figure.dpi'] = 100
    sns.set_style("whitegrid")
    np.random.seed(42)
    
    print(f"Using {CONFIG['n_cores']} CPU cores for parallel processing")
    if CONFIG['use_gpu']:
        try:
            import cupy as cp
            print("GPU acceleration enabled with CuPy")
            return cp
        except ImportError:
            print("CuPy not available, using CPU only")
            CONFIG['use_gpu'] = False
    return np

def load_and_preprocess_eeg(file_path, log_file):
    """Load and preprocess EEG data from EDF file."""
    print("Loading EEG data...")
    log_file.write(f"Loading EEG data from: {file_path}\n")
    
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    log_file.write(f"Original data: {len(raw.ch_names)} channels, {raw.times[-1]:.1f}s duration, {raw.info['sfreq']} Hz\n")
    
    # Filter EEG channels only
    eeg_channels = []
    exclude_patterns = ['Pg', 'REF', 'GND', 'ECG', 'EOG', 'EMG', 'TRIG']
    
    for ch in raw.ch_names:
        if not any(pattern in ch.upper() for pattern in exclude_patterns):
            eeg_channels.append(ch)
    
    raw.pick_channels(eeg_channels)
    log_file.write(f"Selected EEG channels ({len(eeg_channels)}): {', '.join(eeg_channels)}\n")
    
    # Resample if needed
    if raw.info['sfreq'] != CONFIG['sampling_freq']:
        raw.resample(CONFIG['sampling_freq'])
        log_file.write(f"Resampled to {CONFIG['sampling_freq']} Hz\n")
    
    data = raw.get_data()
    times = raw.times
    
    return data, eeg_channels, times, raw.info['sfreq']

def apply_frequency_filter(data, freq_band, sampling_freq, xp=np):
    """Apply bandpass filter for specific frequency band."""
    low_freq, high_freq = freq_band
    
    # Design filter
    nyquist = sampling_freq / 2
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)  # Ensure high freq is below Nyquist
    
    # Use scipy signal processing
    sos = signal.butter(4, [low, high], btype='band', output='sos')
    
    if CONFIG['use_gpu'] and xp.__name__ == 'cupy':
        # GPU processing
        data_gpu = xp.asarray(data)
        filtered_data = xp.zeros_like(data_gpu)
        for i in range(data.shape[0]):
            filtered_data[i] = xp.asarray(signal.sosfilt(sos, xp.asnumpy(data_gpu[i])))
        return xp.asnumpy(filtered_data)
    else:
        # CPU processing
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i] = signal.sosfilt(sos, data[i])
        return filtered_data

def create_time_windows(data_length, sampling_freq):
    """Create overlapping time windows."""
    window_samples = int(CONFIG['window_length'] * sampling_freq)
    overlap_samples = int(window_samples * CONFIG['overlap_ratio'])
    step_samples = window_samples - overlap_samples
    
    windows = []
    start = 0
    while start + window_samples <= data_length:
        windows.append((start, start + window_samples))
        start += step_samples
    
    return windows

def granger_causality_single_pair(args):
    """Calculate Granger causality for a single channel pair."""
    data_x, data_y, max_lag = args
    try:
        # Combine data for GC test
        test_data = np.column_stack([data_y, data_x])  # [target, source]
        
        # Run Granger causality test
        result = grangercausalitytests(test_data, max_lag, verbose=False)
        
        # Get best result across all lags
        best_pvalue = min(result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1))
        best_fstat = min(result[lag][0]['ssr_ftest'][0] for lag in range(1, max_lag + 1) 
                        if result[lag][0]['ssr_ftest'][1] == best_pvalue)
        
        return best_fstat, best_pvalue
    
    except Exception as e:
        return 0.0, 1.0

def calculate_granger_causality_window(data, channels):
    """Calculate Granger causality for all pairs in a time window."""
    n_channels = len(channels)
    gc_matrix = np.zeros((n_channels, n_channels))
    p_matrix = np.ones((n_channels, n_channels))
    
    # Prepare all pairs for parallel processing
    pairs = []
    pair_indices = []
    
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                pairs.append((data[i], data[j], CONFIG['max_lag']))
                pair_indices.append((i, j))
    
    # Process pairs in parallel
    with ProcessPoolExecutor(max_workers=CONFIG['n_cores']) as executor:
        results = list(executor.map(granger_causality_single_pair, pairs))
    
    # Fill matrices
    for (i, j), (fstat, pval) in zip(pair_indices, results):
        gc_matrix[i, j] = fstat
        p_matrix[i, j] = pval
    
    return gc_matrix, p_matrix

def analyze_frequency_band(data, channels, freq_band_name, freq_band, sampling_freq, log_file):
    """Analyze single frequency band with time windows."""
    print(f"\nAnalyzing {freq_band_name} band ({freq_band[0]}-{freq_band[1]} Hz)...")
    log_file.write(f"\n=== {freq_band_name.upper()} BAND ANALYSIS ({freq_band[0]}-{freq_band[1]} Hz) ===\n")
    
    # Filter data to frequency band
    filtered_data = apply_frequency_filter(data, freq_band, sampling_freq)
    
    # Create time windows
    windows = create_time_windows(filtered_data.shape[1], sampling_freq)
    log_file.write(f"Created {len(windows)} time windows with {CONFIG['overlap_ratio']*100}% overlap\n")
    
    # Store results for all windows
    all_connections = defaultdict(int)
    window_results = []
    
    print(f"Processing {len(windows)} time windows...")
    for win_idx, (start, end) in enumerate(windows):
        if win_idx % 10 == 0:
            print(f"  Window {win_idx+1}/{len(windows)}")
        
        # Extract window data
        window_data = filtered_data[:, start:end]
        
        # Calculate Granger causality for this window
        gc_matrix, p_matrix = calculate_granger_causality_window(window_data, channels)
        
        # Find significant connections
        significant_mask = p_matrix < CONFIG['alpha']
        significant_pairs = np.where(significant_mask)
        
        # Store significant connections
        window_connections = []
        for i, j in zip(significant_pairs[0], significant_pairs[1]):
            source = channels[i]
            target = channels[j]
            connection = f"{source}->{target}"
            all_connections[connection] += 1
            window_connections.append({
                'source': source,
                'target': target,
                'fstat': gc_matrix[i, j],
                'pvalue': p_matrix[i, j]
            })
        
        window_results.append({
            'window': win_idx,
            'start_time': start / sampling_freq,
            'end_time': end / sampling_freq,
            'connections': window_connections,
            'n_significant': len(window_connections)
        })
    
    log_file.write(f"Found {len(all_connections)} unique connections across all windows\n")
    
    return all_connections, window_results

def create_network_graph(connections_dict, channels, freq_band_name, output_dir):
    """Create network graph showing top connections."""
    if not connections_dict:
        print(f"No connections found for {freq_band_name} band")
        return None
    
    # Convert to Counter and get top connections
    connections_counter = Counter(connections_dict)
    top_connections = connections_counter.most_common(CONFIG['top_connections'])
    
    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(channels)
    
    # Add edges with weights
    max_count = top_connections[0][1] if top_connections else 1
    for connection, count in top_connections:
        source, target = connection.split('->')
        G.add_edge(source, target, weight=count, normalized_weight=count/max_count)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw all nodes in one color
    nx.draw_networkx_nodes(G, pos, node_color='lightcoral', node_size=1000, alpha=0.8)
    
    # Draw edges with consistent style
    nx.draw_networkx_edges(G, pos, edge_color='darkred', alpha=0.6, 
                          width=2, arrows=True, arrowsize=20)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title(f'{freq_band_name.title()} Band - Top {CONFIG["top_connections"]} Connections\n'
              f'(Edge thickness = frequency of significant connection)')
    plt.axis('off')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f'{freq_band_name}_network.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return G

def create_connectivity_heatmap(all_bands_connections, channels, output_dir):
    """Create heatmap showing brain region connectivity across all bands."""
    n_channels = len(channels)
    
    # Create connectivity matrix for each band
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    band_names = list(CONFIG['frequency_bands'].keys())
    
    for idx, (band_name, connections) in enumerate(all_bands_connections.items()):
        if idx >= len(axes):
            break
            
        # Create connectivity matrix
        conn_matrix = np.zeros((n_channels, n_channels))
        
        for connection, count in connections.items():
            source, target = connection.split('->')
            if source in channels and target in channels:
                i = channels.index(source)
                j = channels.index(target)
                conn_matrix[i, j] = count
        
        # Plot heatmap
        sns.heatmap(conn_matrix, 
                   xticklabels=channels, 
                   yticklabels=channels,
                   cmap='Reds', 
                   ax=axes[idx],
                   cbar_kws={'label': 'Connection Frequency'})
        
        axes[idx].set_title(f'{band_name.title()} Band')
        axes[idx].set_xlabel('Target')
        axes[idx].set_ylabel('Source')
    
    # Remove empty subplots
    for idx in range(len(all_bands_connections), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Brain Region Connectivity Across Frequency Bands', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'connectivity_heatmap_all_bands.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_connections_csv(all_bands_connections, output_dir):
    """Save all connections to CSV files."""
    for band_name, connections in all_bands_connections.items():
        # Create DataFrame
        connections_data = []
        for connection, frequency in connections.items():
            source, target = connection.split('->')
            connections_data.append({
                'source': source,
                'target': target,
                'frequency': frequency,
                'band': band_name
            })
        
        df = pd.DataFrame(connections_data)
        df = df.sort_values('frequency', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f'{band_name}_connections.csv')
        df.to_csv(csv_path, index=False)
    
    # Save combined CSV
    all_data = []
    for band_name, connections in all_bands_connections.items():
        for connection, frequency in connections.items():
            source, target = connection.split('->')
            all_data.append({
                'source': source,
                'target': target,
                'frequency': frequency,
                'band': band_name
            })
    
    combined_df = pd.DataFrame(all_data)
    combined_df = combined_df.sort_values(['band', 'frequency'], ascending=[True, False])
    combined_df.to_csv(os.path.join(output_dir, 'all_connections.csv'), index=False)

def create_summary_statistics(all_bands_connections, channels, log_file):
    """Create summary statistics for all frequency bands."""
    log_file.write("\n" + "="*50 + "\n")
    log_file.write("SUMMARY STATISTICS ACROSS ALL FREQUENCY BANDS\n")
    log_file.write("="*50 + "\n")
    
    for band_name, connections in all_bands_connections.items():
        log_file.write(f"\n{band_name.upper()} BAND:\n")
        log_file.write(f"  Total unique connections: {len(connections)}\n")
        
        if connections:
            total_occurrences = sum(connections.values())
            avg_frequency = total_occurrences / len(connections)
            max_frequency = max(connections.values())
            
            log_file.write(f"  Total connection occurrences: {total_occurrences}\n")
            log_file.write(f"  Average connection frequency: {avg_frequency:.2f}\n")
            log_file.write(f"  Maximum connection frequency: {max_frequency}\n")
            
            # Top 5 connections
            top_5 = Counter(connections).most_common(5)
            log_file.write(f"  Top 5 connections:\n")
            for i, (conn, freq) in enumerate(top_5, 1):
                log_file.write(f"    {i}. {conn}: {freq} times\n")

def setup_output_directory(edf_path):
    """Create subject-specific output directory."""
    filename = os.path.basename(edf_path).replace('.edf', '')
    
    # Extract subject info from BIDS format if available
    if 'sub-' in filename:
        parts = filename.split('_')
        subject_parts = [p for p in parts if p.startswith('sub-') or p.startswith('ses-')]
        subject_id = '_'.join(subject_parts) if subject_parts else filename
    else:
        subject_id = filename
    
    output_dir = os.path.join('.', 'frequency_band_analysis_output', subject_id)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def main():
    """Main analysis function."""
    if len(sys.argv) != 2:
        print("Usage: python individual_frequency_bands_analysis.py <path_to_edf_file>")
        sys.exit(1)
    
    edf_file = sys.argv[1]
    if not os.path.exists(edf_file):
        print(f"Error: File {edf_file} not found")
        sys.exit(1)
    
    # Setup
    xp = setup_environment()
    output_dir = setup_output_directory(edf_file)
    
    # Open log file
    log_path = os.path.join(output_dir, 'analysis_log.txt')
    with open(log_path, 'w') as log_file:
        log_file.write(f"FREQUENCY BAND EEG ANALYSIS LOG\n")
        log_file.write(f"File: {edf_file}\n")
        log_file.write(f"Analysis started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Configuration: {CONFIG}\n\n")
        
        print(f"Starting analysis of: {edf_file}")
        print(f"Output directory: {output_dir}")
        
        # Load and preprocess data
        data, channels, times, sampling_freq = load_and_preprocess_eeg(edf_file, log_file)
        
        # Analyze each frequency band
        all_bands_connections = {}
        
        for band_name, freq_range in CONFIG['frequency_bands'].items():
            connections, window_results = analyze_frequency_band(
                data, channels, band_name, freq_range, sampling_freq, log_file
            )
            all_bands_connections[band_name] = connections
            
            # Create network graph for this band
            create_network_graph(connections, channels, band_name, output_dir)
        
        # Create combined visualizations
        print("\nCreating combined visualizations...")
        create_connectivity_heatmap(all_bands_connections, channels, output_dir)
        
        # Save results to CSV
        print("Saving results to CSV files...")
        save_connections_csv(all_bands_connections, output_dir)
        
        # Create summary statistics
        create_summary_statistics(all_bands_connections, channels, log_file)
        
        log_file.write(f"\nAnalysis completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Log file: {log_path}")

if __name__ == "__main__":
    main()
