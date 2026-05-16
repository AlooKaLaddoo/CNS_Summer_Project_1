import csv
import json

import networkx as nx
import numpy as np


def main():
	A = np.array([
		[0.0, 0.43708611, 0.95564288, 0.75879455, 0.63879264, 0.24041678, 0.24039507, 0.15227525, 0.87955853, 0.64100351],
		[0.43708611, 0.0, 0.73726532, 0.11852604, 0.97291887, 0.84919838, 0.2911052, 0.26364247, 0.26506406, 0.37381802],
		[0.95564288, 0.73726532, 0.0, 0.57228079, 0.48875052, 0.36210623, 0.65066761, 0.22554447, 0.36293018, 0.42972566],
		[0.75879455, 0.11852604, 0.57228079, 0.0, 0.51046299, 0.80665837, 0.2797064, 0.56281099, 0.63317311, 0.14180537],
		[0.63879264, 0.97291887, 0.48875052, 0.51046299, 0.0, 0.64679037, 0.25347171, 0.15854643, 0.95399698, 0.96906883],
		[0.24041678, 0.84919838, 0.36210623, 0.80665837, 0.64679037, 0.0, 0.82755761, 0.37415239, 0.1879049, 0.71580972],
		[0.24039507, 0.2911052, 0.65066761, 0.2797064, 0.25347171, 0.82755761, 0.0, 0.49613724, 0.20983441, 0.54565922],
		[0.15227525, 0.26364247, 0.22554447, 0.56281099, 0.15854643, 0.37415239, 0.49613724, 0.0, 0.13094967, 0.91838836],
		[0.87955853, 0.26506406, 0.36293018, 0.63317311, 0.95399698, 0.1879049, 0.20983441, 0.13094967, 0.0, 0.33290198],
		[0.64100351, 0.37381802, 0.42972566, 0.14180537, 0.96906883, 0.71580972, 0.54565922, 0.91838836, 0.33290198, 0.0],
	], dtype=float)

	edge_length = 1.0 / A
	np.fill_diagonal(edge_length, 0.0)

	G = nx.from_numpy_array(edge_length)

	node_bc = nx.betweenness_centrality(G, normalized=True, weight="weight")
	edge_bc = nx.edge_betweenness_centrality(G, normalized=True, weight="weight")

	n = A.shape[0]
	nodes = [float(node_bc.get(i, 0.0)) for i in range(n)]

	edges = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] > 0]
	edge_list = [
		{"u": int(u), "v": int(v), "weight": float(A[u, v]), "betweenness": float(edge_bc.get((u, v), edge_bc.get((v, u), 0.0)))}
		for (u, v) in edges
	]

	out = {"nodes": nodes, "edges": edge_list}
	print(json.dumps(out, indent=2))

	nodes_csv = "betweenness_nodes_weighted.csv"
	edges_csv = "betweenness_edges_weighted.csv"

	with open(nodes_csv, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["node", "betweenness"])
		for i, val in enumerate(nodes):
			writer.writerow([i, val])

	with open(edges_csv, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["u", "v", "weight", "betweenness"])
		for e in edge_list:
			writer.writerow([e["u"], e["v"], e["weight"], e["betweenness"]])

	print(f"Wrote {nodes_csv} and {edges_csv}")


if __name__ == "__main__":
	main()
