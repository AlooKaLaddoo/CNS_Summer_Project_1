#!/usr/bin/env python3
"""
EEG Granger Causality Analysis Tool
Usage: python granger_causality_analysis.py <path_to_edf_file>
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
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Configuration - Modify these parameters as needed
CONFIG = {
    'lowpass': 30,          # High-pass filter frequency (Hz)
    'highpass': 0.5,        # Low-pass filter frequency (Hz)  
    'target_fs': 100,       # Target sampling frequency (Hz)
    'max_lag': 2,           # Maximum lag for Granger causality
    'alpha': 0.05,          # Significance threshold
    'window_length': 10,    # Time window length (seconds)
    'overlap_ratio': 0.3,   # Window overlap ratio (0-1)
    'threshold_percentile': 95,  # Network graph threshold percentile
    'max_pairs': 5,         # Max pairs for temporal dynamics analysis
}

def setup_environment():
    """Initialize environment settings and multiprocessing"""
    warnings.filterwarnings('ignore')
    mne.set_log_level('WARNING')
    plt.rcParams['figure.dpi'] = 100
    sns.set_palette("husl")
    np.random.seed(42)
    
    n_cores = mp.cpu_count()
    print(f"Detected {n_cores} CPU cores - using all for parallel processing")
    return n_cores

def load_eeg_data(file_path):
    """Load EEG data from EDF file"""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    print(f"Loaded EEG data: {len(raw.ch_names)} channels, {raw.times[-1]:.1f}s duration")
    return raw

def setup_output_folder(file_path):
    """Create output folder based on filename"""
    filename = os.path.basename(file_path).replace('.edf', '')
    output_dir = f'./Dataset/Infants_data_output/{filename}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output folder: {output_dir}")
    return output_dir

def preprocess_eeg(raw):
    """Clean and filter EEG data"""
    raw_filtered = raw.copy()
    raw_filtered.filter(CONFIG['highpass'], CONFIG['lowpass'], fir_design='firwin', verbose=False)
    data, times = raw_filtered[:, :]
    
    # Standard EEG electrodes
    eeg_names = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'FZ', 'C3', 'C4', 'CZ', 
                 'P3', 'P4', 'PZ', 'O1', 'O2', 'T3', 'T4', 'T5', 'T6']
    
    eeg_indices = []
    eeg_channels = []
    
    for i, ch in enumerate(raw_filtered.ch_names):
        if ch in eeg_names:
            eeg_indices.append(i)
            eeg_channels.append(ch)
    
    clean_data = data[eeg_indices, :]
    print(f"Preprocessed: {len(eeg_channels)} EEG channels")
    return clean_data, eeg_channels, times

def plot_eeg_sample(data, channels, times, sampling_freq, save_path=None):
    """Plot EEG sample data for visualization"""
    duration = min(10, len(times) / sampling_freq)  # Plot up to 10 seconds
    samples = int(duration * sampling_freq)
    time_slice = slice(0, min(samples, data.shape[1]))
    
    n_channels = min(4, len(channels))  # Plot first 4 channels
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    for i in range(n_channels):
        axes[i].plot(times[time_slice], data[i, time_slice])
        axes[i].set_title(f'{channels[i]}')
        axes[i].set_ylabel('µV')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'EEG Sample Data ({duration:.1f} seconds)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def prepare_data(data, original_fs):
    """Downsample and prepare data for analysis"""
    data = np.asarray(data)
    
    if original_fs <= CONFIG['target_fs']:
        processed_data = data.copy()
        new_fs = original_fs
    else:
        n_new = int(data.shape[1] * CONFIG['target_fs'] / original_fs)
        processed_data = np.asarray(signal.resample(data, n_new, axis=1))
        new_fs = CONFIG['target_fs']
        print(f"Downsampled: {original_fs} → {new_fs} Hz")
    
    print(f"Using all data: {processed_data.shape}")
    return processed_data, new_fs

def calculate_granger_causality_pair(args):
    """Helper function for parallel GC computation"""
    i, j, data, channel_names, max_lag = args
    try:
        test_data = np.column_stack([data[j, :], data[i, :]])
        result = grangercausalitytests(test_data, max_lag, verbose=False)
        best_result = min(
            (result[lag][0]['ssr_ftest'] for lag in range(1, max_lag + 1)),
            key=lambda x: x[1]
        )
        return i, j, best_result[0], best_result[1]
    except Exception:
        return i, j, 0.0, 1.0

def calculate_granger_causality(data, channel_names, n_cores):
    """Calculate Granger causality between all channel pairs"""
    n_channels = len(channel_names)
    gc_matrix = np.zeros((n_channels, n_channels))
    p_values = np.ones((n_channels, n_channels))
    
    print(f"Calculating Granger causality for {n_channels} channels using {n_cores} cores...")
    
    pairs = [(i, j, data, channel_names, CONFIG['max_lag']) 
             for i in range(n_channels) for j in range(n_channels) if i != j]
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(calculate_granger_causality_pair, pairs))
    
    for i, j, f_stat, p_val in results:
        gc_matrix[i, j] = f_stat
        p_values[i, j] = p_val
    
    return gc_matrix, p_values

def apply_significance_threshold(gc_matrix, p_values):
    """Apply significance threshold to GC matrix"""
    return np.where(p_values <= CONFIG['alpha'], gc_matrix, 0)

def plot_connectivity_matrix(matrix, channel_names, title, save_path=None):
    """Plot connectivity matrix as heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, 
                xticklabels=channel_names, 
                yticklabels=channel_names,
                cmap='hot', 
                cbar_kws={'label': 'GC F-statistic'},
                square=True)
    
    plt.title(title)
    plt.xlabel('Target (Y)')
    plt.ylabel('Source (X)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_connectivity_network(gc_matrix, channel_names, save_path=None):
    """Plot connectivity as a network graph"""
    non_zero_values = gc_matrix[gc_matrix > 0]
    if len(non_zero_values) == 0:
        print("No connections found in the matrix")
        return None
    
    threshold = np.percentile(non_zero_values, CONFIG['threshold_percentile'])
    thresholded_matrix = np.where(gc_matrix >= threshold, gc_matrix, 0)
    G = nx.from_numpy_array(thresholded_matrix, create_using=nx.DiGraph)
    
    node_mapping = {i: name for i, name in enumerate(channel_names)}
    G = nx.relabel_nodes(G, node_mapping)
    
    if G.number_of_edges() == 0:
        print(f"No edges found with threshold percentile {CONFIG['threshold_percentile']}")
        return G
    
    plt.figure(figsize=(12, 12))
    pos = nx.circular_layout(G)
    
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [weight / max_weight * 3 for weight in edge_weights]
    
    nx.draw(G, pos, 
            node_color='lightblue', 
            node_size=1000,
            with_labels=True,
            font_size=10,
            font_weight='bold',
            edge_color='red',
            arrows=True,
            arrowsize=20,
            width=edge_widths,
            alpha=0.7)
    
    plt.title(f'EEG Connectivity Network (Top {100-CONFIG["threshold_percentile"]}% connections)')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def analyze_connectivity_statistics(gc_matrix, p_values, channel_names, save_dir=None):
    """Analyze connectivity statistics and create visualizations"""
    significant_mask = p_values < CONFIG['alpha']
    n_total = len(channel_names) * (len(channel_names) - 1)
    n_significant = np.sum(significant_mask)
    
    print(f"CONNECTIVITY SUMMARY")
    print(f"Significant: {n_significant}/{n_total} ({n_significant/n_total*100:.1f}%)")
    
    out_degree = np.sum(significant_mask, axis=1)
    in_degree = np.sum(significant_mask, axis=0)
    
    node_stats = pd.DataFrame({
        'Channel': channel_names,
        'Out': out_degree,
        'In': in_degree,
        'Total': out_degree + in_degree
    }).sort_values('Total', ascending=False)
    
    print(f"\nTOP CONNECTED CHANNELS")
    print(node_stats.head(10).to_string(index=False))
    
    connections_df = None
    if n_significant > 0:
        sig_i, sig_j = np.where(significant_mask)
        connections_df = pd.DataFrame({
            'Source': [channel_names[i] for i in sig_i],
            'Target': [channel_names[j] for j in sig_j],
            'F-stat': gc_matrix[sig_i, sig_j],
            'p-value': p_values[sig_i, sig_j]
        }).sort_values('F-stat', ascending=False)
        
        print(f"\nTOP 10 STRONGEST CONNECTIONS")
        print(connections_df.head(10).to_string(index=False))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(node_stats['Out'], node_stats['In'], s=100, alpha=0.7, color='steelblue')
    for _, row in node_stats.iterrows():
        ax1.annotate(row['Channel'], (row['Out'], row['In']), 
                    xytext=(3, 3), textcoords='offset points', fontsize=9)
    ax1.set_xlabel('Out-degree')
    ax1.set_ylabel('In-degree')
    ax1.set_title('Channel Connectivity Pattern')
    ax1.grid(True, alpha=0.3)
    
    top10 = node_stats.head(10)
    ax2.barh(range(len(top10)), top10['Total'], color='orange', alpha=0.7)
    ax2.set_yticks(range(len(top10)))
    ax2.set_yticklabels(top10['Channel'])
    ax2.set_xlabel('Total Degree')
    ax2.set_title('Top 10 Most Connected Channels')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'connectivity_statistics.png'), 
                   dpi=300, bbox_inches='tight')
    plt.close()
    
    return node_stats, connections_df

def calculate_region_connectivity(region_pair_args):
    """Helper for parallel region connectivity calculation"""
    i, j, src_region, tgt_region, channel_names, gc_matrix, p_values, ch_to_region, alpha = region_pair_args
    
    strengths = []
    for src_ch in channel_names:
        for tgt_ch in channel_names:
            if (src_ch in ch_to_region and tgt_ch in ch_to_region and
                ch_to_region[src_ch] == src_region and 
                ch_to_region[tgt_ch] == tgt_region and 
                src_ch != tgt_ch):
                
                src_idx, tgt_idx = channel_names.index(src_ch), channel_names.index(tgt_ch)
                if p_values[src_idx, tgt_idx] < alpha:
                    strengths.append(gc_matrix[src_idx, tgt_idx])
    
    return i, j, np.mean(strengths) if strengths else 0

def analyze_brain_regions(gc_matrix, p_values, channel_names, n_cores, save_path=None):
    """Analyze connectivity patterns by brain regions"""
    regions = {
        'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'FZ'],
        'Central': ['C3', 'C4', 'CZ'],
        'Parietal': ['P3', 'P4', 'PZ'],
        'Occipital': ['O1', 'O2'],
        'Temporal': ['T3', 'T4', 'T5', 'T6']
    }
    
    ch_to_region = {ch: region for region, channels in regions.items() 
                    for ch in channels if ch in channel_names}
    
    region_names = list(regions.keys())
    region_matrix = np.zeros((len(region_names), len(region_names)))
    
    region_args = []
    for i, src_region in enumerate(region_names):
        for j, tgt_region in enumerate(region_names):
            region_args.append((i, j, src_region, tgt_region, channel_names, 
                              gc_matrix, p_values, ch_to_region, CONFIG['alpha']))
    
    print(f"Calculating region connectivity")
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(calculate_region_connectivity, region_args))
    
    for i, j, strength in results:
        region_matrix[i, j] = strength
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(region_matrix, 
                xticklabels=region_names, 
                yticklabels=region_names,
                annot=True, fmt='.2f', cmap='hot',
                cbar_kws={'label': 'Mean GC Strength'})
    
    plt.title('Inter-Region Connectivity\n(Mean Granger Causality Strength)')
    plt.xlabel('Target Region')
    plt.ylabel('Source Region')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nBRAIN REGION CONNECTIVITY SUMMARY")
    for i, src in enumerate(region_names):
        for j, tgt in enumerate(region_names):
            if region_matrix[i, j] > 0:
                print(f"{src} → {tgt}: {region_matrix[i, j]:.3f}")
    
    return region_matrix

def segment_data_into_windows(data, sampling_freq):
    """Segment EEG data into overlapping time windows"""
    window_samples = int(CONFIG['window_length'] * sampling_freq)
    step_samples = int(window_samples * (1 - CONFIG['overlap_ratio']))
    
    windows = []
    window_info = []
    
    for start_idx in range(0, data.shape[1] - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples
        
        windows.append(data[:, start_idx:end_idx])
        window_info.append({
            'window_idx': len(windows) - 1,
            'start_time': start_idx / sampling_freq,
            'end_time': end_idx / sampling_freq,
            'duration': CONFIG['window_length']
        })
    
    print(f"Created {len(windows)} windows of {CONFIG['window_length']}s each (overlap: {CONFIG['overlap_ratio']*100:.0f}%)")
    
    return windows, window_info

def calculate_gc_for_window(args):
    """Helper for parallel window processing"""
    window_idx, window_data, channel_names, n_cores = args
    try:
        gc_matrix, p_values = calculate_granger_causality(window_data, channel_names, n_cores)
        significant_gc = apply_significance_threshold(gc_matrix, p_values)
        return window_idx, gc_matrix, p_values, significant_gc
    except Exception:
        n_channels = len(channel_names)
        zero_matrix = np.zeros((n_channels, n_channels))
        one_matrix = np.ones((n_channels, n_channels))
        return window_idx, zero_matrix, one_matrix, zero_matrix

def analyze_time_windows(data, channel_names, sampling_freq, n_cores):
    """Calculate Granger causality across multiple time windows"""
    windows, window_info = segment_data_into_windows(data, sampling_freq)
    
    n_channels = len(channel_names)
    gc_results = {
        'gc_matrices': [None] * len(windows),
        'p_value_matrices': [None] * len(windows),
        'significant_matrices': [None] * len(windows),
        'window_info': window_info
    }
    
    print(f"Calculating GC for {len(windows)} windows")
    
    window_args = [(i, window_data, channel_names, 1)  # Use 1 core per window to avoid nested parallelism
                   for i, window_data in enumerate(windows)]
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(calculate_gc_for_window, window_args))
    
    for window_idx, gc_matrix, p_values, significant_gc in results:
        gc_results['gc_matrices'][window_idx] = gc_matrix
        gc_results['p_value_matrices'][window_idx] = p_values
        gc_results['significant_matrices'][window_idx] = significant_gc
        
        n_connections = np.sum(significant_gc > 0)
        print(f"Window {window_idx+1}/{len(windows)}: ✓ {n_connections} connections")
    
    return gc_results

def plot_connectivity_evolution(gc_results, channel_names, save_path=None):
    """Plot how connectivity evolves across time windows"""
    gc_matrices = gc_results['gc_matrices']
    window_info = gc_results['window_info']
    
    metrics = []
    for i, gc_matrix in enumerate(gc_matrices):
        total_conn = np.sum(gc_matrix > 0)
        mean_strength = np.mean(gc_matrix[gc_matrix > 0]) if total_conn > 0 else 0
        
        metrics.append({
            'start_time': window_info[i]['start_time'],
            'total_connections': total_conn,
            'mean_strength': mean_strength,
            'max_strength': np.max(gc_matrix)
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    ax1.plot(metrics_df['start_time'], metrics_df['mean_strength'], 'o-', color='orange', linewidth=2)
    ax1.set_title('Connectivity Evolution Across Time Windows')
    ax1.set_ylabel('Mean GC Strength')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(metrics_df['start_time'], metrics_df['max_strength'], 'o-', color='red', linewidth=2)
    ax2.set_ylabel('Max GC Strength')
    ax2.set_xlabel('Time (seconds)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics_df

def analyze_connectivity_stability(gc_results, channel_names, save_path=None):
    """Analyze stability of connectivity patterns across time windows"""
    significant_matrices = gc_results['significant_matrices']
    n_windows = len(significant_matrices)
    n_channels = len(channel_names)
    
    consistency = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                consistency[i, j] = sum(matrix[i, j] > 0 for matrix in significant_matrices) / n_windows
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(consistency, 
                xticklabels=channel_names, 
                yticklabels=channel_names,
                cmap='viridis', 
                vmin=0, vmax=1,
                cbar_kws={'label': 'Connection Consistency'})
    
    plt.title(f'Connection Consistency Across {n_windows} Windows')
    plt.xlabel('Target Channel')
    plt.ylabel('Source Channel')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    stable_list = []
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j and consistency[i, j] > 0.5:
                stable_list.append({
                    'source': channel_names[i],
                    'target': channel_names[j],
                    'consistency': consistency[i, j]
                })
    
    if stable_list:
        stable_df = pd.DataFrame(stable_list).sort_values('consistency', ascending=False)
        print(f"\nMOST STABLE CONNECTIONS ({len(stable_list)} found)")
        print(stable_df.head(15).to_string(index=False))
        return consistency, stable_df
    else:
        print("\nNo highly stable connections found (>50% consistency)")
        return consistency, None

def save_results(output_dir, gc_matrix, p_values, significant_gc, channel_names, connections_df=None, node_stats=None, gc_results=None):
    """Save all analysis results"""
    saved_files = []
    
    # Save numpy arrays
    for name, data in [('gc_matrix', gc_matrix), ('p_values', p_values), ('significant_gc', significant_gc)]:
        np.save(os.path.join(output_dir, f'{name}.npy'), data)
        saved_files.append(f'{name}.npy')
    
    # Save window analysis if available
    if gc_results:
        np.save(os.path.join(output_dir, 'window_matrices.npy'), gc_results['gc_matrices'])
        np.save(os.path.join(output_dir, 'window_pvalues.npy'), gc_results['p_value_matrices'])
        saved_files.extend(['window_matrices.npy', 'window_pvalues.npy'])
    
    # Save CSV files
    if connections_df is not None:
        connections_df.to_csv(os.path.join(output_dir, 'connections.csv'), index=False)
        saved_files.append('connections.csv')
        
    if node_stats is not None:
        node_stats.to_csv(os.path.join(output_dir, 'node_stats.csv'), index=False)
        saved_files.append('node_stats.csv')
    
    # Save channel names
    with open(os.path.join(output_dir, 'channels.txt'), 'w') as f:
        f.write('\n'.join(channel_names))
    saved_files.append('channels.txt')
    
    print(f"Saved {len(saved_files)} files to: {output_dir}")

def main():
    """Main analysis pipeline"""
    if len(sys.argv) != 2:
        print("Usage: python granger_causality_analysis.py <path_to_edf_file>")
        sys.exit(1)
    
    edf_file = sys.argv[1]
    if not os.path.exists(edf_file):
        print(f"Error: File {edf_file} not found")
        sys.exit(1)
    
    print("=== EEG Granger Causality Analysis ===")
    
    # 1. Setup
    n_cores = setup_environment()
    
    # 2. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    raw = load_eeg_data(edf_file)
    output_dir = setup_output_folder(edf_file)
    eeg_data, channel_names, times = preprocess_eeg(raw)
    sampling_freq = raw.info['sfreq']
    
    # Plot sample data
    plot_eeg_sample(eeg_data, channel_names, times, sampling_freq, 
                   os.path.join(output_dir, 'eeg_sample_data.png'))
    
    # Prepare data for analysis
    eeg_subset, new_fs = prepare_data(eeg_data, sampling_freq)
    
    # 3. Calculate Granger causality
    print("\n2. Calculating Granger causality...")
    gc_matrix, p_values = calculate_granger_causality(eeg_subset, channel_names, n_cores)
    significant_gc = apply_significance_threshold(gc_matrix, p_values)
    
    n_sig, n_total = np.sum(significant_gc > 0), len(channel_names) * (len(channel_names) - 1)
    print(f"Found {n_sig}/{n_total} significant connections ({n_sig/n_total*100:.1f}%)")
    
    # 4. Create visualizations
    print("\n3. Creating visualizations...")
    plot_connectivity_matrix(gc_matrix, channel_names, 
                            'Granger Causality Matrix (All Connections)',
                            os.path.join(output_dir, 'gc_matrix_all_connections.png'))
    
    plot_connectivity_matrix(significant_gc, channel_names,
                            'Significant Granger Causality Connections (p < 0.05)',
                            os.path.join(output_dir, 'gc_matrix_significant_connections.png'))
    
    G = plot_connectivity_network(significant_gc, channel_names,
                                 os.path.join(output_dir, 'connectivity_network.png'))
    
    # 5. Statistical analysis
    print("\n4. Statistical analysis...")
    node_stats, connections_df = analyze_connectivity_statistics(
        gc_matrix, p_values, channel_names, output_dir)
    
    # 6. Brain region analysis
    print("\n5. Brain region analysis...")
    region_matrix = analyze_brain_regions(
        gc_matrix, p_values, channel_names, n_cores,
        os.path.join(output_dir, 'brain_region_connectivity.png'))
    
    # 7. Time window analysis
    print("\n6. Time window analysis...")
    gc_results = analyze_time_windows(eeg_subset, channel_names, new_fs, n_cores)
    
    metrics_df = plot_connectivity_evolution(
        gc_results, channel_names,
        os.path.join(output_dir, 'connectivity_evolution.png'))
    
    consistency_matrix, stable_connections = analyze_connectivity_stability(
        gc_results, channel_names,
        os.path.join(output_dir, 'connectivity_stability.png'))
    
    # 8. Save results
    print("\n7. Saving results...")
    save_results(output_dir, gc_matrix, p_values, significant_gc, channel_names, 
                 connections_df, node_stats, gc_results)
    
    print("\n✓ ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
