# EEG Channel Correlation Analysis

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

# Set plotting parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

print("Libraries imported successfully!")


# Batch processing of all .edf files in ./Dataset/Infants_data
import os
import glob

input_dir = "./Dataset/Infants_data"
output_base = "./Dataset/Correlation_output"
os.makedirs(output_base, exist_ok=True)

edf_files = glob.glob(os.path.join(input_dir, "**", "*.edf"), recursive=True)

def get_top_correlations(corr_matrix, n=10):
    """Get top n correlations (excluding self-correlations)"""
    upper_tri = np.triu_indices_from(corr_matrix, k=1)
    correlations = corr_matrix.values[upper_tri]
    top_indices = np.argsort(correlations)[-n:][::-1]
    bottom_indices = np.argsort(correlations)[:n]
    top_pairs = [(upper_tri[0][i], upper_tri[1][i]) for i in top_indices]
    bottom_pairs = [(upper_tri[0][i], upper_tri[1][i]) for i in bottom_indices]
    return top_pairs, bottom_pairs, correlations[top_indices], correlations[bottom_indices]

for edf_file_path in edf_files:
    edf_filename = os.path.basename(edf_file_path)
    subject_name = os.path.splitext(edf_filename)[0]
    output_dir = os.path.join(output_base, subject_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing {edf_file_path}")
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Channel names: {raw.ch_names}")
    print(f"Recording duration: {raw.times[-1]:.2f} seconds")
    print(f"Total samples: {len(raw.times)}")

    print("Original data shape:", raw.get_data().shape)
    raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names if not ch.startswith('Pg')})
    raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=30.0, verbose=False)
    eeg_channels = [ch for ch in raw_filtered.ch_names if not ch.startswith('Pg')]
    raw_eeg = raw_filtered.pick_channels(eeg_channels)
    print(f"EEG channels for analysis: {raw_eeg.ch_names}")
    print(f"Filtered data shape: {raw_eeg.get_data().shape}")
    eeg_data = raw_eeg.get_data()  # Shape: (channels, time_points)
    channel_names = raw_eeg.ch_names
    sfreq = raw_eeg.info['sfreq']
    print(f"Final data for correlation analysis:")
    print(f"- Channels: {len(channel_names)}")
    print(f"- Time points: {eeg_data.shape[1]}")
    print(f"- Duration: {eeg_data.shape[1]/sfreq:.1f} seconds")

    print("Computing correlation matrix...")
    eeg_data_transposed = eeg_data.T
    df_eeg = pd.DataFrame(eeg_data_transposed, columns=channel_names)
    correlation_matrix = df_eeg.corr(method='pearson')
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    print(f"Matrix values range: {correlation_matrix.values.min():.3f} to {correlation_matrix.values.max():.3f}")
    corr_values = correlation_matrix.values
    off_diagonal = corr_values[~np.eye(corr_values.shape[0], dtype=bool)]
    print(f"\nCorrelation statistics (excluding diagonal):")
    print(f"Mean correlation: {off_diagonal.mean():.3f}")
    print(f"Std correlation: {off_diagonal.std():.3f}")
    print(f"Min correlation: {off_diagonal.min():.3f}")
    print(f"Max correlation: {off_diagonal.max():.3f}")

    # Save correlation matrix as CSV
    correlation_matrix.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))

    # Visualization 1: Correlation Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                cbar_kws={"shrink": .8},
                annot_kws={'size': 6})
    plt.title('EEG Channel Correlation Matrix (Complete)', fontsize=16, fontweight='bold')
    plt.xlabel('EEG Channels', fontsize=12)
    plt.ylabel('EEG Channels', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

    # Visualization 2: Distribution of Correlation Values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(off_diagonal, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(off_diagonal.mean(), color='red', linestyle='--', label=f'Mean: {off_diagonal.mean():.3f}')
    ax.set_title('Distribution of Channel Correlations')
    ax.set_xlabel('Correlation Coefficient')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_histogram.png"))
    plt.close()

    # Save statistics to a text file
    with open(os.path.join(output_dir, "correlation_stats.txt"), "w") as f:
        f.write(f"Mean: {off_diagonal.mean():.3f}\n")
        f.write(f"Std: {off_diagonal.std():.3f}\n")
        f.write(f"Min: {off_diagonal.min():.3f}\n")
        f.write(f"Max: {off_diagonal.max():.3f}\n")
        f.write(f"Median: {np.median(off_diagonal):.3f}\n")

    # Visualization 3: Top and Bottom Correlations Analysis
    top_pairs, bottom_pairs, top_values, bottom_values = get_top_correlations(correlation_matrix, n=10)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top_labels = [f"{channel_names[i]}-{channel_names[j]}" for i, j in top_pairs]
    axes[0].barh(range(len(top_values)), top_values, color='darkgreen', alpha=0.7)
    axes[0].set_yticks(range(len(top_values)))
    axes[0].set_yticklabels(top_labels)
    axes[0].set_xlabel('Correlation Coefficient')
    axes[0].set_title('Top 10 Highest Channel Correlations')
    axes[0].grid(True, alpha=0.3)
    for i, v in enumerate(top_values):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')
    bottom_labels = [f"{channel_names[i]}-{channel_names[j]}" for i, j in bottom_pairs]
    axes[1].barh(range(len(bottom_values)), bottom_values, color='darkred', alpha=0.7)
    axes[1].set_yticks(range(len(bottom_values)))
    axes[1].set_yticklabels(bottom_labels)
    axes[1].set_xlabel('Correlation Coefficient')
    axes[1].set_title('Top 10 Lowest Channel Correlations')
    axes[1].grid(True, alpha=0.3)
    for i, v in enumerate(bottom_values):
        axes[1].text(v - 0.01, i, f'{v:.3f}', va='center', ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_bottom_correlations.png"))
    plt.close()

    # Save top/bottom correlation pairs to text file
    with open(os.path.join(output_dir, "top_bottom_pairs.txt"), "w") as f:
        f.write("Highest correlations:\n")
        for (i, j), corr in zip(top_pairs, top_values):
            f.write(f"{channel_names[i]} - {channel_names[j]}: {corr:.3f}\n")
        f.write("\nLowest correlations:\n")
        for (i, j), corr in zip(bottom_pairs, bottom_values):
            f.write(f"{channel_names[i]} - {channel_names[j]}: {corr:.3f}\n")

    # Visualization 4: Regional Analysis and Network Visualization
    import matplotlib.patches as mpatches
    brain_regions = {
        'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'FZ'],
        'Central': ['C3', 'C4', 'CZ'],
        'Parietal': ['P3', 'P4', 'PZ'],
        'Temporal': ['T3', 'T4', 'T5', 'T6'],
        'Occipital': ['O1', 'O2']
    }
    region_map = {}
    for region, channels in brain_regions.items():
        for ch in channels:
            if ch in channel_names:
                region_map[ch] = region
    within_region_corrs = {}
    between_region_corrs = {}
    for region in brain_regions:
        region_channels = [ch for ch in brain_regions[region] if ch in channel_names]
        if len(region_channels) > 1:
            region_indices = [channel_names.index(ch) for ch in region_channels]
            region_corr_matrix = correlation_matrix.iloc[region_indices, region_indices]
            mask = np.triu(np.ones_like(region_corr_matrix, dtype=bool), k=1)
            within_region_corrs[region] = region_corr_matrix.values[mask]
    region_names = list(brain_regions.keys())
    for i, region1 in enumerate(region_names):
        for region2 in region_names[i+1:]:
            channels1 = [ch for ch in brain_regions[region1] if ch in channel_names]
            channels2 = [ch for ch in brain_regions[region2] if ch in channel_names]
            if channels1 and channels2:
                indices1 = [channel_names.index(ch) for ch in channels1]
                indices2 = [channel_names.index(ch) for ch in channels2]
                between_corrs = []
                for idx1 in indices1:
                    for idx2 in indices2:
                        between_corrs.append(correlation_matrix.iloc[idx1, idx2])
                between_region_corrs[f"{region1}-{region2}"] = between_corrs
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    within_means = {region: np.mean(corrs) for region, corrs in within_region_corrs.items()}
    regions = list(within_means.keys())
    means = list(within_means.values())
    axes[0,0].bar(regions, means, color=['red', 'blue', 'green', 'orange', 'purple'][:len(regions)], alpha=0.7)
    axes[0,0].set_title('Mean Within-Region Correlations')
    axes[0,0].set_ylabel('Mean Correlation')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    for i, v in enumerate(means):
        axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    between_means = {pair: np.mean(corrs) for pair, corrs in between_region_corrs.items()}
    pairs = list(between_means.keys())
    between_vals = list(between_means.values())
    axes[0,1].barh(range(len(pairs)), between_vals, color='darkblue', alpha=0.7)
    axes[0,1].set_yticks(range(len(pairs)))
    axes[0,1].set_yticklabels(pairs)
    axes[0,1].set_title('Mean Between-Region Correlations')
    axes[0,1].set_xlabel('Mean Correlation')
    axes[0,1].grid(True, alpha=0.3)
    all_within = np.concatenate(list(within_region_corrs.values()))
    all_between = np.concatenate(list(between_region_corrs.values()))
    axes[1,0].hist([all_within, all_between], bins=30, alpha=0.7, 
                   label=['Within-region', 'Between-region'], color=['red', 'blue'])
    axes[1,0].set_title('Distribution: Within vs Between Region Correlations')
    axes[1,0].set_xlabel('Correlation Coefficient')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    data_to_plot = [all_within, all_between]
    axes[1,1].boxplot(data_to_plot, labels=['Within-region', 'Between-region'])
    axes[1,1].set_title('Box Plot: Within vs Between Region Correlations')
    axes[1,1].set_ylabel('Correlation Coefficient')
    axes[1,1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regional_analysis.png"))
    plt.close()

    # Save regional analysis stats
    with open(os.path.join(output_dir, "regional_stats.txt"), "w") as f:
        f.write(f"Mean within-region correlation: {np.mean(all_within):.3f} ± {np.std(all_within):.3f}\n")
        f.write(f"Mean between-region correlation: {np.mean(all_between):.3f} ± {np.std(all_between):.3f}\n")
        f.write(f"Difference: {np.mean(all_within) - np.mean(all_between):.3f}\n")

    # Save summary
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("EEG CHANNEL CORRELATION ANALYSIS SUMMARY\n")
        f.write(f"Subject: {subject_name}\n")
        f.write(f"Channels analyzed: {len(channel_names)}\n")
        f.write(f"Recording duration: {eeg_data.shape[1]/sfreq:.1f} seconds\n")
        f.write(f"Sampling frequency: {sfreq} Hz\n")
        f.write(f"Matrix size: {correlation_matrix.shape[0]} × {correlation_matrix.shape[1]}\n")
        f.write(f"Total channel pairs: {len(off_diagonal)}\n")
        f.write(f"Mean correlation: {off_diagonal.mean():.3f}\n")
        f.write(f"Standard deviation: {off_diagonal.std():.3f}\n")
        f.write(f"Range: [{off_diagonal.min():.3f}, {off_diagonal.max():.3f}]\n")
        f.write(f"1. Highest correlation: {off_diagonal.max():.3f}\n")
        f.write(f"2. Lowest correlation: {off_diagonal.min():.3f}\n")
        if 'all_within' in locals() and 'all_between' in locals():
            f.write(f"3. Within-region correlations: {np.mean(all_within):.3f} ± {np.std(all_within):.3f}\n")
            f.write(f"4. Between-region correlations: {np.mean(all_between):.3f} ± {np.std(all_between):.3f}\n")
        f.write("Note: This analysis shows the functional connectivity patterns between EEG channels in resting-state infant brain activity.\n")
        f.write("Final Correlation Matrix:\n")
        f.write(correlation_matrix.round(3).to_string())

print("Batch processing complete.")
