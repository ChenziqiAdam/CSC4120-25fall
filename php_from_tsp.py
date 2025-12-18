import networkx as nx
from mtsp_dp import mtsp_dp
from student_utils import *

def php_solver_from_tsp(G, H):
    """
    PHP solver via reduction to Euclidean TSP.

    Parameters:
        G (nx.Graph): A NetworkX graph representing the city.
            This directed graph is equivalent to an undirected one by construction.
        H (list): A list of home nodes that must be visited.

    Returns:
        list: A list of nodes traversed by your car (the computed tour).

    Notes:
        - All nodes are represented as integers.
        - Solve the problem by first transforming the PTHP problem to a TSP problem.
        - Use the dynamic programming algorithm introduced in lectures to solve TSP.
        - Construct a solution for the original PTHP problem after solving TSP.

    Constraints:
        - The tour must begin and end at node 0.
        - The tour can only traverse existing edges in the graph.
        - The tour must visit every node in H.
    """

    if not G or 0 not in G.nodes():
        return []

    # Ensure unique required nodes and exclude 0 from H if it appears.
    required_nodes = [0]
    for h in H:
        if h != 0 and h not in required_nodes:
            required_nodes.append(h)

    # Trivial case: only the house exists.
    if len(required_nodes) == 1:
        return [0]

    # Build the reduced complete graph on homes + node 0 with edge weights as
    # shortest-path distances in the original graph.
    all_pairs_dist = nx.floyd_warshall(G, weight='weight')
    reduced_graph = nx.Graph()
    reduced_graph.add_nodes_from(required_nodes)

    for i, u in enumerate(required_nodes):
        for v in required_nodes[i + 1:]:
            reduced_graph.add_edge(u, v, weight=all_pairs_dist[u][v])

    tsp_tour = mtsp_dp(reduced_graph)
    if not tsp_tour:
        return []

    # Expand each edge of the TSP tour back to the corresponding shortest path
    # in the original graph.
    expanded_tour = []
    for i in range(len(tsp_tour) - 1):
        u, v = tsp_tour[i], tsp_tour[i + 1]
        segment = nx.shortest_path(G, source=u, target=v, weight='weight')
        if expanded_tour:
            expanded_tour.extend(segment[1:])  # avoid duplicating the junction
        else:
            expanded_tour.extend(segment)

    # Ensure the tour ends at 0 even in degenerate cases.
    if expanded_tour and expanded_tour[-1] != 0:
        expanded_tour.append(0)

    return expanded_tour


if __name__ == "__main__":
    pass
