import json
import numpy as np
import networkx as nx


def main():
	A = np.array([
		[0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		[0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
		[1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
		[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
		[0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
	], dtype=int)

	G = nx.from_numpy_array(A)

	# Node and edge betweenness (unweighted, normalized)
	node_bc = nx.betweenness_centrality(G, normalized=True)
	edge_bc = nx.edge_betweenness_centrality(G, normalized=True)

	n = A.shape[0]
	nodes = [float(node_bc.get(i, 0.0)) for i in range(n)]

	edges = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 1]
	edge_list = [
		{"u": int(u), "v": int(v), "betweenness": float(edge_bc.get((u, v), edge_bc.get((v, u), 0.0)))}
		for (u, v) in edges
	]

	out = {"nodes": nodes, "edges": edge_list}
	print(json.dumps(out, indent=2))

	# Save CSVs
	import csv

	nodes_csv = "betweenness_nodes.csv"
	edges_csv = "betweenness_edges.csv"

	with open(nodes_csv, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["node", "betweenness"])
		for i, val in enumerate(nodes):
			writer.writerow([i, val])

	with open(edges_csv, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["u", "v", "betweenness"])
		for e in edge_list:
			writer.writerow([e["u"], e["v"], e["betweenness"]])

	print(f"Wrote {nodes_csv} and {edges_csv}")


if __name__ == "__main__":
	main()

