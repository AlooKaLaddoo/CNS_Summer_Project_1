# Multi-Session Jaccard Index Analysis

## Overview

This document explains the multi-session Jaccard Index analysis performed on infant EEG data to assess the **temporal stability** and **reliability** of brain connectivity patterns across different recording sessions.

---

## What is Multi-Session Analysis?

When infants undergo multiple EEG recording sessions (e.g., at different ages or on different days), we can compare their brain connectivity patterns between sessions to understand:

1. **Test-Retest Reliability**: Are measurements consistent when repeated?
2. **Temporal Stability**: Do connectivity patterns remain stable over time?
3. **Developmental Changes**: How do networks evolve as infants mature?

---

## Methodology

### Step 1: Data Identification
- **Input**: Preprocessed EEG data from all subjects (`.npy` files + metadata)
- **Process**: Scan the preprocessing summary to identify subjects with multiple sessions
- **Output**: List of multi-session subjects (e.g., sub-NORB00069 with 4 sessions, sub-NORB00064 with 3 sessions)

### Step 2: Channel Alignment
Since different sessions may have different numbers of channels (due to bad channel removal), we:
- Identify **common channels** present in both sessions
- Extract only these channels for fair comparison
- Example: If session 1 has 19 channels and session 2 has 17 channels, we use only the channels common to both

### Step 3: Correlation Matrix Computation
For each session:
1. **Average each epoch** across time → get mean activity per channel per epoch
2. **Compute correlation matrix** between all channel pairs → (n_channels × n_channels) matrix
3. **Remove self-correlations** (diagonal = 0)

This gives us a correlation matrix showing how each brain region co-activates with every other region.

### Step 4: Network Binarization (Thresholding)
- Convert correlation matrices to **binary adjacency matrices** using multiple thresholds (0.3, 0.4, 0.5, 0.6, 0.7)
- Edges exist where `|correlation| >= threshold`
- Testing multiple thresholds ensures results aren't artifacts of a single cutoff

### Step 5: Jaccard Index Calculation
The **Jaccard Index** measures similarity between two networks:

```
Jaccard Index = Intersection / Union
              = (Common edges) / (Total unique edges)
```

- **High Jaccard (0.7-1.0)**: Networks are very similar → stable/reliable connectivity
- **Moderate Jaccard (0.4-0.7)**: Some consistency, some variability
- **Low Jaccard (0.0-0.4)**: Networks are very different → high variability or developmental change

### Step 6: Multi-Threshold Analysis
By computing Jaccard at 5 different thresholds, we assess:
- **Robustness**: Do results hold across different correlation cutoffs?
- **Threshold sensitivity**: Are differences due to actual network changes or just threshold artifacts?

---

## What the Code Does

### Cell-by-Cell Breakdown

#### Cells 1-4: Setup
- Import libraries (NumPy, Pandas, Matplotlib, Seaborn)
- Define paths and thresholds
- Create output directory

#### Cell 5: Identify Multi-Session Subjects
```python
# Finds subjects with more than 1 session
session_counts = summary_df.groupby('subject_id').size()
multi_session_subjects = session_counts[session_counts > 1]['subject_id']
```
**Output**: List of subjects like sub-NORB00096, sub-NORB00087, sub-NORB00069, etc.

#### Cell 7: Helper Functions
- `load_preprocessed_epochs()`: Load .npy data and metadata
- `compute_correlation_matrix()`: Calculate channel-to-channel correlations
- `threshold_matrix()`: Convert correlations to binary (0/1) adjacency matrix
- `jaccard_index()`: Calculate intersection/union of two networks
- `align_channels()`: Match channels between sessions

#### Cell 9: Main Analysis Loop
For each multi-session subject:
1. Get all session pairs (e.g., ses-1 vs ses-2, ses-1 vs ses-3)
2. Load both sessions' data
3. Align channels
4. Compute correlation matrices for both
5. For each threshold (0.3-0.7):
   - Binarize networks
   - Calculate Jaccard Index
   - Count edges in each network
6. Store results

#### Cells 11-18: Visualization
- **Distribution histograms**: Show Jaccard values at each threshold
- **Box plots**: Compare threshold sensitivity
- **Per-subject bars**: Identify which subjects have stable/variable networks
- **Correlation heatmaps**: Visual comparison of actual connectivity patterns

#### Cell 20: Key Findings Summary
Automated interpretation based on:
- Mean Jaccard across all comparisons
- Threshold-specific patterns
- Best/worst similarity pairs
- Overall stability assessment

---

## Expected Findings & Interpretation

### High Jaccard Index (>0.6)
**Interpretation**: 
- Networks are **highly stable** across sessions
- Measurements are **reliable**
- Connectivity patterns are consistent

**Implications**:
- Good quality data
- True biological stability (if sessions are close in time)
- Robust network structure

### Moderate Jaccard Index (0.4-0.6)
**Interpretation**:
- **Some stability** but also variability
- Partial network reorganization
- May reflect state changes (drowsy vs alert)

**Implications**:
- Normal for developmental studies
- Consider session timing and infant state
- Look at which specific connections change

### Low Jaccard Index (<0.4)
**Interpretation**:
- **High variability** between sessions
- Networks are substantially different

**Possible reasons**:
1. **Data quality issues**: Different artifact levels, bad channel differences
2. **State differences**: Sleep vs wake, deep sleep vs light sleep
3. **Developmental changes**: Real maturation (if sessions are months apart)
4. **Threshold too strict**: Networks have few edges, small changes cause big Jaccard drops

---

## Advantages of This Analysis

### 1. **Multi-Threshold Robustness**
- Not dependent on arbitrary single threshold choice
- Can identify if low Jaccard is due to network sparsity vs actual differences

### 2. **Channel Alignment**
- Handles different channel counts between sessions
- Fair comparison using only shared channels

### 3. **Comprehensive Metrics**
- Stores edge counts, epoch counts, channel counts
- Enables post-hoc investigation of low-similarity cases

### 4. **Visual Diagnostics**
- Heatmaps show actual connectivity patterns
- Can identify specific regions that change

---

## Limitations & Considerations

### 1. **Small Sample Size**
- Only ~10-15 subjects typically have multiple sessions in infant datasets
- Limited statistical power

### 2. **Threshold Dependence**
Even with multiple thresholds, binarization loses information:
- Correlation of 0.49 → no edge
- Correlation of 0.51 → edge
- Small changes near threshold cause large Jaccard changes

**Alternative**: Use **weighted Jaccard** or directly compare correlation matrices

### 3. **Epoch Count Differences**
- Session 1: 20 epochs
- Session 2: 15 epochs
- Different sample sizes affect correlation stability

### 4. **Time Between Sessions Unknown**
- Sessions 1 day apart should have high Jaccard
- Sessions 6 months apart may have low Jaccard (developmental change is expected)
- Analysis doesn't account for inter-session interval

### 5. **State Confounds**
Infant EEG is highly state-dependent:
- "Eyes closed" can range from drowsy to deep sleep
- Even clean segments may have different arousal levels
- Low Jaccard might reflect state, not poor reliability

---

## Output Files

The analysis generates:

1. **`multi_session_jaccard_results.csv`**
   - One row per subject-session pair-threshold combination
   - Columns: subject_id, session1, session2, threshold, jaccard_index, n_edges, etc.

2. **`jaccard_summary_by_threshold.csv`**
   - Summary statistics (mean, median, std, min, max) per threshold
   - Shows overall trends

3. **`jaccard_distribution_by_threshold.png`**
   - Histograms showing Jaccard distributions at each threshold

4. **`jaccard_vs_threshold_boxplot.png`**
   - Box plots comparing all thresholds side-by-side

5. **`per_subject_jaccard_comparison.png`**
   - Bar chart showing each subject's Jaccard (at threshold=0.5)

6. **`correlation_heatmap_[subject].png`**
   - Side-by-side heatmaps of actual correlation matrices for example subject

---

## How to Interpret Your Results

### If Mean Jaccard > 0.6:
✅ **Good news**: Networks are stable and reliable
- Your measurements are consistent
- Connectivity patterns are robust
- Data quality is likely good

### If Mean Jaccard = 0.4-0.6:
⚠️ **Moderate stability**: Some consistency, some change
- **Check**: Time between sessions
- **Check**: Infant states (sleep stages)
- **Consider**: Developmental changes if sessions are months apart

### If Mean Jaccard < 0.4:
🔍 **Investigate further**:
1. Look at per-subject breakdown (some infants stable, others not?)
2. Check edge counts (sparse networks = unstable Jaccard)
3. Review correlation heatmaps (visual inspection)
4. Consider data quality differences between sessions
5. Try weighted Jaccard or direct correlation comparison

---

## Comparison to Inter-Subject Analysis

**Important baseline**: Compare multi-session Jaccard to **inter-subject Jaccard** (comparing different babies):

- **Within-subject (multi-session)** Jaccard should be **higher** than **between-subject** Jaccard
- If multi-session Jaccard ≈ inter-subject Jaccard → networks are as different within a baby as between babies → **poor reliability**
- If multi-session Jaccard >> inter-subject Jaccard → networks are stable within a baby → **good reliability**

**Next analysis**: Run Jaccard between randomly selected subject pairs at session 1 to establish this baseline.

---

## Recommendations for Future Analysis

### 1. **Add Time Information**
- Extract session dates from metadata
- Calculate days/weeks between sessions
- Plot Jaccard vs inter-session interval
- Expected: Jaccard decreases as time increases (developmental change)

### 2. **Weighted Jaccard**
Instead of binary edges, use correlation values:
```python
weighted_jaccard = sum(min(corr1, corr2)) / sum(max(corr1, corr2))
```
This is less sensitive to threshold artifacts.

### 3. **Direct Correlation Comparison**
- Flatten correlation matrices to vectors
- Compute Pearson correlation between vectors
- Gives continuous similarity measure without thresholding

### 4. **Graph Metrics Comparison**
Compare higher-level network properties:
- Clustering coefficient
- Path length
- Modularity
- Degree distribution

If these are similar even when edges differ, core network structure may be stable.

### 5. **Subnetwork Analysis**
- Compare anterior vs posterior regions separately
- Interhemispheric vs intrahemispheric connections
- Identify which specific connections are stable/variable

---

## Biological Interpretation

### High Multi-Session Jaccard Suggests:
1. **Trait-like connectivity**: Core network architecture is stable
2. **Reliable measurements**: EEG captures consistent biological signals
3. **Mature networks**: If present in older infants

### Low Multi-Session Jaccard Might Indicate:
1. **State-dependent networks**: Connectivity varies with arousal/sleep stage
2. **Developmental plasticity**: Networks reorganizing rapidly (expected in young infants)
3. **Measurement noise**: Poor data quality or preprocessing inconsistencies

### Developmental Context:
- **0-6 months**: Rapid synaptogenesis, expect **lower** stability
- **6-12 months**: Network consolidation, expect **higher** stability
- **Inter-session interval matters**: 1 week vs 3 months have very different expectations

---

## Conclusion

This multi-session Jaccard analysis provides a **quantitative assessment of network stability** across recording sessions. By using multiple thresholds, channel alignment, and comprehensive visualizations, we can:

1. Assess **test-retest reliability** of EEG connectivity measures
2. Identify **stable vs variable** subjects
3. Detect potential **data quality issues**
4. Understand **developmental changes** when sessions span significant time

The analysis is most informative when:
- Combined with inter-subject baseline comparison
- Interpreted in light of inter-session timing
- Supplemented with visual inspection of correlation matrices
- Considered alongside infant age and state information

**Key takeaway**: Jaccard Index is a simple but powerful metric for comparing brain networks across time, but must be interpreted carefully considering data quality, infant state, and developmental context.
