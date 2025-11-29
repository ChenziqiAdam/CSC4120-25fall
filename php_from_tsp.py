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
    
    # reduction
    tsp_tour = mtsp_dp(reduced_graph)
    # reduction

    return tour


if __name__ == "__main__":
    pass