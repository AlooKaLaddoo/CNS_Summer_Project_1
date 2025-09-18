# Participation Coefficient in Brain Network Analysis

## What is Participation Coefficient?
The participation coefficient is a graph-theoretical measure that quantifies how evenly a node (e.g., an EEG channel or brain region) connects to different communities (modules) within a network. In brain network analysis, it helps identify whether a node acts as a connector between different modules or is mainly connected within its own module.

- **High participation coefficient:** Node connects broadly across many modules (integrative hub).
- **Low participation coefficient:** Node connects mostly within its own module (provincial hub).

## How is it Calculated?
1. **Data Loading:** The notebook loads a correlation matrix representing functional connectivity between EEG channels.
2. **Network Construction:** A graph is built where nodes are EEG channels and edges are weighted by correlation values.
3. **Community Detection:** A community detection algorithm (e.g., Louvain) is used to assign each node to a module.
4. **Participation Coefficient Calculation:** For each node, the proportion of its connections to each module is computed using:
   
   $$P_i = 1 - \sum_{s=1}^{M} \left( \frac{k_{is}}{k_i} \right)^2$$
   - $k_{is}$: Number of edges from node $i$ to nodes in module $s$
   - $k_i$: Total degree of node $i$
   - $M$: Number of modules

5. **Visualization:** The participation coefficients are visualized as a bar plot or histogram to show the distribution across channels.

## Interpreting Results
- Channels with high participation coefficients are likely to be important for inter-module communication.
- Channels with low coefficients are more specialized within their module.

This analysis helps understand the integrative and specialized roles of EEG channels in infant brain networks.