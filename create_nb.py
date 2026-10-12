import nbformat as nbf

nb = nbf.v4.new_notebook()

text = """# Node Degree Analysis
This notebook loads the segment-wise correlation matrices and computes the binary node degree using a specified threshold."""

code1 = """import os
import pandas as pd
import numpy as np
from tqdm import tqdm

base_dir = '/home/aloo/CNS_Summer_Project_1/correlation_matrices_segment_wise_(notAbsolute)'
segment_matrices = {}

# Iterate over all subjects
subjects = sorted([d for d in os.listdir(base_dir) if d.startswith('sub-')])

for subject in tqdm(subjects, desc="Loading subjects"):
    subject_path = os.path.join(base_dir, subject)
    ses_path = os.path.join(subject_path, 'ses-1')
    
    if os.path.isdir(ses_path):
        segment_matrices[subject] = {}
        for file in os.listdir(ses_path):
            if file.endswith('.csv'):
                # Extract segment name, e.g., 'segment-01'
                parts = file.split('_')
                segment = next((p for p in parts if p.startswith('segment-')), None)
                
                if segment:
                    file_path = os.path.join(ses_path, file)
                    # Load the correlation matrix
                    df = pd.read_csv(file_path, index_col=0)
                    segment_matrices[subject][segment] = df

print(f"Successfully loaded correlation matrices for {len(segment_matrices)} subjects.")"""

code2 = """# Calculate Node Degree using Thresholding
threshold = 0.3  # You can adjust this threshold as needed
node_degrees_segment = {}
rows = []

for subject, segments in tqdm(segment_matrices.items(), desc="Computing node degrees"):
    node_degrees_segment[subject] = {}
    for segment, df in segments.items():
        # Use absolute correlation values
        abs_df = df.abs()
        
        # Create a binary adjacency matrix based on the threshold
        adj_matrix = (abs_df > threshold).astype(int)
        
        # Set diagonal to 0 to avoid counting self-connections
        np.fill_diagonal(adj_matrix.values, 0)
        
        # Sum the rows to get the degree of each node
        degrees = adj_matrix.sum(axis=1)
        node_degrees_segment[subject][segment] = degrees.to_dict()
        
        # Flatten into rows for a DataFrame
        for node, degree in degrees.items():
            rows.append({
                'subject': subject,
                'segment': segment,
                'node': node,
                'degree': degree
            })

node_degree_df = pd.DataFrame(rows)
print(f"Computed binary node degrees for all segments. DataFrame shape: {node_degree_df.shape}")

# Display the first few rows
node_degree_df.head()"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2)
]

with open('/home/aloo/CNS_Summer_Project_1/degreeAnalysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully.")
