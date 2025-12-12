# Understanding Jaccard Index for EEG Brain Connectivity

## Table of Contents
1. [What is the Jaccard Index?](#what-is-the-jaccard-index)
2. [How It Works with EEG Data](#how-it-works-with-eeg-data)
3. [Why Use Correlation? (Pros & Cons)](#why-use-correlation-pros--cons)
4. [Analysis Workflow](#analysis-workflow)
5. [Understanding Your Results](#understanding-your-results)
6. [References](#references)

---

## What is the Jaccard Index?

Think of the Jaccard Index as a way to measure "how similar are two things?" It's like comparing two friends' favorite movies:

- If they like 8 of the same movies out of 10 total unique movies between them: Jaccard = 8/10 = 0.8 (very similar!)
- If they only share 2 movies out of 10 total: Jaccard = 2/10 = 0.2 (not very similar)

**In simple terms:**
- **Score of 1.0**: Perfectly identical
- **Score of 0.5**: Half similar, half different  
- **Score of 0.0**: Completely different

**Why it's useful for brain data:**
Instead of comparing movie lists, we compare brain connection patterns. This helps us answer questions like:
- Does this infant's brain connectivity stay stable over time?
- Do two infants have similar brain organization?
- How does brain connectivity change as the baby grows?

---

## How It Works with EEG Data

### The Process in Plain Language

**Step 1: Measure Brain Activity**  
Your EEG device records electrical signals from 19 sensors on the infant's head (like Fp1, F3, C3, etc.).

**Step 2: Find Connections Between Brain Regions**  
We calculate how synchronized each pair of sensors is. If two sensors show similar patterns, they're "connected."

**Step 3: Create a Connection Map**  
This gives us a correlation matrix - a table showing how strongly each sensor pair is connected (values from -1 to +1).

**Step 4: Simplify to Yes/No Connections**  
We pick a threshold (like 0.5) and convert to binary:
- Correlation ≥ 0.5 → Connection EXISTS (1)
- Correlation < 0.5 → NO connection (0)

This creates a simple network of which brain regions talk to each other.

**Step 5: Compare Two Networks**  
Use Jaccard Index to see how similar two networks are by counting:
- Shared connections (in both networks)
- Unique connections (only in one network)

**Formula:**
$$\text{Jaccard} = \frac{\text{Shared connections}}{\text{All unique connections}}$$

---

## Why Use Correlation? (Pros & Cons)

### Why Correlation is a GOOD Choice

 **Easy to Understand**  
Correlation is intuitive - it measures how two brain signals move together over time.

 **Widely Used**  
Most brain connectivity studies use correlation, making your results comparable to other research.

 **Computationally Simple**  
Fast to calculate, even with long recordings and many channels.

 **Captures Linear Relationships**  
Works well for detecting synchronized brain activity patterns.

 **No Assumptions About Direction**  
Correlation doesn't assume one region "causes" activity in another - it just measures co-activity.

### Why Correlation Might Be a BAD Choice

 **Loses Information**  
When you threshold correlations to 0/1, you lose details. A correlation of 0.51 and 0.99 both become "1" (connected).

 **Arbitrary Thresholds**  
Choosing 0.5 as your cutoff is somewhat arbitrary. Different thresholds give different results.

 **Only Linear Relationships**  
Correlation misses complex, non-linear brain interactions. Two regions might interact in sophisticated ways that correlation can't detect.

 **Sensitive to Artifacts**  
Eye movements, muscle activity, or noise can create false correlations between channels.

 **Reference Dependency**  
EEG correlation depends on how you reference your electrodes (common reference, average reference, etc.).

 **Volume Conduction**  
Because electrical signals spread through the skull, nearby electrodes may appear correlated just due to physical proximity, not true brain connectivity.

### Better Alternatives to Consider

1. **Coherence**: Measures frequency-specific synchronization (better than simple correlation)
2. **Phase Locking Value**: Captures phase synchronization without amplitude effects
3. **Granger Causality**: Tests if one region's activity predicts another's (directional)
4. **Mutual Information**: Detects non-linear relationships

### When Correlation + Jaccard Works Well

Despite limitations, this approach is excellent for:
- **Exploratory analysis** - getting a first look at connectivity patterns
- **Temporal stability** - tracking if patterns stay consistent over time
- **Group comparisons** - identifying infants with similar brain organization
- **Quick screening** - rapid analysis of large datasets

**Bottom Line:** Correlation + Jaccard is a good starting point, but consider it one tool among many. For deeper insights, combine it with other methods.

---

## Analysis Workflow

### What Can You Analyze?

**1. Does Brain Connectivity Stay Stable Over Time?**  
Split one recording into 10-second windows and compare:
- Window 1 vs Window 2
- Window 2 vs Window 3, etc.
- High Jaccard = stable brain state
- Low Jaccard = changing brain patterns

**2. Which Infants Have Similar Brain Organization?**  
Compare connectivity between all pairs of infants:
- High Jaccard = similar brain networks
- Groups with high similarity might be at similar developmental stages

**3. How Does Connectivity Change as Baby Grows?**  
For infants with multiple recordings over time:
- Session 1 vs Session 2 vs Session 3
- Increasing Jaccard = brain networks stabilizing (maturation)
- Decreasing Jaccard = brain reorganization

**4. Do Different Brain States Have Different Connectivity?**  
Compare eyes-open vs eyes-closed periods:
- Shows how brain connectivity responds to environmental changes

**5. Which Frequency Bands Show Similar Connectivity?**  
Compare delta waves (0.5-4 Hz) vs alpha waves (8-13 Hz):
- Reveals which brain rhythms have related connectivity patterns

---

## How to Calculate It (Simple Flow)

### Overview of the Process

**Input:** Two EEG recordings (from same or different infants)  
**Output:** A number from 0 to 1 showing how similar their brain connectivity is

### Step-by-Step Flow

**Step 1: Load EEG Data**
- Read the .edf file
- Extract the 19 EEG channels (Fp1, F3, C3, etc.)
- Get the signal data (channels × time points)

**Step 2: Calculate Correlation Matrix**
- For each pair of channels, measure how synchronized they are
- Creates a 19×19 table of correlation values (-1 to +1)
- Example: Correlation between Fp1 and F3 might be 0.65

**Step 3: Convert to Binary Network**
- Pick a threshold (e.g., 0.5)
- If correlation ≥ 0.5 → mark as "connected" (1)
- If correlation < 0.5 → mark as "not connected" (0)
- Now you have a simple network of 1s and 0s

**Step 4: Compare Two Networks**
- Take binary networks from two recordings
- Count shared connections (both have 1)
- Count unique connections (only one has 1)
- Calculate: Shared ÷ (Shared + Unique) = Jaccard Index

**Step 5: Interpret Result**
- Jaccard = 0.8 → Very similar brain connectivity
- Jaccard = 0.5 → Moderately similar
- Jaccard = 0.2 → Very different connectivity patterns

### Key Technical Details

**For Symmetric Networks:**
- Since connection A→B is same as B→A, only count upper triangle of matrix
- This avoids counting each connection twice
- Ignore the diagonal (channel with itself)

**What Gets Counted:**
- ✓ Connections present in both networks (intersection)
- ✓ Connections only in network 1 (difference)
- ✓ Connections only in network 2 (difference)
- ✗ Absent connections in both (NOT counted - this is important!)

**Example Calculation:**
```
Network 1: 50 connections
Network 2: 45 connections
Shared connections: 30
Unique total: 30 + (50-30) + (45-30) = 30 + 20 + 15 = 65
Jaccard Index: 30/65 = 0.46
```

---

## Implementing for Your Infant Dataset

### What You Need

**Libraries:** Python with MNE (for EEG), NumPy, Pandas, Matplotlib  
**Data:** Your .edf files in `./Dataset/Infants_data/`  
**Output:** Results saved to `./Dataset/Jaccard_Analysis_Output/`

### Analysis Pipeline

**Configuration:**
```
Correlation threshold: 0.5 (connections with |r| ≥ 0.5 are "significant")
Time window: 10 seconds (for temporal stability)
Frequency bands: Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz)
```

**Main Analyses You Can Run:**

1. **Temporal Stability** (Do networks stay stable over time?)
   - Split each recording into 10-second windows
   - Calculate correlation network for each window
   - Compare consecutive windows
   - Output: Mean Jaccard score showing stability

2. **Inter-Subject Similarity** (Which infants are similar?)
   - Load all infant recordings
   - Calculate correlation network for each
   - Compare all pairs
   - Output: Similarity matrix (heatmap) showing clusters

3. **Threshold Sensitivity** (Is 0.5 the right threshold?)
   - Try thresholds from 0.1 to 1.0
   - See how Jaccard changes
   - See how network density changes
   - Output: Graphs showing optimal threshold range

### Visualization Outputs

**1. Temporal Stability Plot**
- Line graph: Jaccard scores between consecutive windows
- Heatmap: All window-to-window comparisons
- Shows: If infant's brain state is consistent or changing

**2. Inter-Subject Heatmap**
- Color-coded matrix: Green = similar, Red = different
- Reveals: Groups of infants with similar brain networks
- Useful for: Identifying developmental subgroups

**3. Threshold Sensitivity Curves**
- Top graph: How Jaccard changes with threshold
- Bottom graph: How number of connections changes
- Helps: Choose appropriate threshold for your data

### Example Use Cases

**Use Case 1: Compare Two Specific Infants**
```
Load infant 1 data → Calculate correlations → Threshold at 0.5
Load infant 2 data → Calculate correlations → Threshold at 0.5
Compare networks → Get Jaccard score
Result: 0.65 means moderately similar brain connectivity
```

**Use Case 2: Track One Infant Over Time**
```
Same infant has recordings at: 3 months, 6 months, 9 months
Calculate networks for each session
Compare: Session 1 vs 2, then 2 vs 3
Rising Jaccard = brain networks maturing and stabilizing
```

**Use Case 3: Compare Brain Frequencies**
```
Take one recording
Filter to Delta band (slow waves) → Get network
Filter to Alpha band (moderate waves) → Get network
Filter to Beta band (fast waves) → Get network
Compare networks across frequencies
Shows: Which brain rhythms have similar connectivity patterns
```

---

## Understanding Your Results

### What Do Different Scores Mean?

| Jaccard Score | What It Means | Example |
|---------------|---------------|---------|
| 0.8 - 1.0 | Almost identical networks | Same infant, back-to-back recordings |
| 0.6 - 0.8 | Very similar patterns | Infants at same developmental stage |
| 0.4 - 0.6 | Moderately similar | Some shared patterns, some differences |
| 0.2 - 0.4 | Mostly different | Different developmental stages |
| 0.0 - 0.2 | Completely different | Different brain states or conditions |

### What Results Tell You About...

**Temporal Stability (Same infant, different time windows):**
- **High score (>0.7)**: Brain is in a stable state (good for analysis)
- **Low score (<0.4)**: Brain is transitioning between states or recording has artifacts
- **Moderate (0.4-0.7)**: Normal variability - brain is active but not chaotic

**Comparing Infants:**
- **High score (>0.6)**: Similar brain development
- **Low score (<0.3)**: Individual differences (normal for infants!)
- Remember: Infant brains change rapidly, so lower scores are expected

**Tracking Development Over Months:**
- **Score increasing**: Networks stabilizing (brain maturing)
- **Score decreasing**: Networks reorganizing (developmental changes)
- **Stable scores**: Consistent developmental trajectory

### Things That Change Your Results

**1. Your Threshold Choice (e.g., 0.5)**
- Too high (like 0.9): Very few connections detected, results unstable
- Too low (like 0.2): Too many connections, everything looks similar
- Sweet spot: Usually 0.4-0.6 for EEG data

**2. How Dense Your Network Is**
- More connections → Higher Jaccard scores (naturally more overlap)
- Fewer connections → Lower Jaccard scores (harder to match)
- Always report how many connections you found!

**3. Recording Length**
- Longer recordings → More reliable correlations → Trustworthy Jaccard
- Short windows (<5 seconds) → Noisy correlations → Unreliable Jaccard
- Recommendation: At least 10 seconds per window

**4. Data Quality**
- Clean data → Consistent patterns → Meaningful Jaccard scores
- Noisy data (infant moving, crying) → Random patterns → Misleading scores
- Tip: Visually inspect your correlation matrices first!

### Is Your Result Statistically Significant?

**Simple Check:** Compare against chance
- Shuffle one network randomly
- Calculate Jaccard with shuffled version
- Repeat 1000 times
- If your real Jaccard is higher than 95% of shuffled versions → Significant!

**Example:**
```
Real Jaccard: 0.65
Average shuffled Jaccard: 0.30
→ Your result is meaningful (not due to chance)
```

### Common Mistakes to Avoid

 **Comparing different age groups directly** - Expect low Jaccard, doesn't mean bad data  
 **Using one threshold only** - Try multiple (0.3, 0.5, 0.7) to verify  
 **Ignoring network density** - Always check how many connections you have  
 **Not checking data quality** - One bad electrode can mess up everything  
 **Over-interpreting small differences** - 0.55 vs 0.58 might not be meaningful

---

## References

### Essential Reading

1. **Rubinov, M., & Sporns, O. (2010)**. "Complex network measures of brain connectivity." *NeuroImage*, 52(3), 1059-1069.
   - Complete guide to brain network analysis including Jaccard Index
   - DOI: 10.1016/j.neuroimage.2009.10.003

2. **Bosch-Bayard, J., et al. (2022)**. "EEG effective connectivity during the first year of life." *NeuroImage*, 252, 119035.
   - **Your dataset!** - Shows how connectivity changes in infant brains
   - DOI: 10.1016/j.neuroimage.2022.119035

3. **Bullmore, E., & Sporns, O. (2009)**. "Complex brain networks: graph theoretical analysis." *Nature Reviews Neuroscience*, 10(3), 186-198.
   - Explains why network approaches work for brain data
   - DOI: 10.1038/nrn2575

4. **Zalesky, A., et al. (2012)**. "On the use of correlation as a measure of network connectivity." *NeuroImage*, 60(4), 2096-2106.
   - Critical paper on correlation's pros and cons
   - DOI: 10.1016/j.neuroimage.2012.02.001

---

## Quick Summary

**What is Jaccard Index?**  
A simple number (0 to 1) that tells you how similar two brain connectivity patterns are.

**Why use it with correlation?**  
✅ Easy to calculate and understand  
✅ Works well for exploratory analysis  
❌ Loses detail when converting to binary  
❌ Threshold choice is somewhat arbitrary  

**Best for:**
- Checking if brain networks stay stable over time
- Finding infants with similar brain organization  
- Tracking developmental changes across sessions
- Quick screening of large datasets

**Tips:**
- Use threshold around 0.4-0.6 for EEG
- Try multiple thresholds to verify results
- Check data quality before calculating
- Report network density alongside Jaccard scores
- Consider using other methods (coherence, phase locking) for deeper analysis

**Your Dataset:**  
103 infants, 130 recordings, 19 channels each - perfect for Jaccard analysis to understand infant brain development patterns!
