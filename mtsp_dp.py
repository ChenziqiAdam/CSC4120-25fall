import networkx as nx
from itertools import combinations

def mtsp_dp(G):
    """
    Solve the Traveling Salesman Problem (TSP) using dynamic programming.

    Parameters:
        G (nx.Graph): A NetworkX graph representing the city.

    Returns:
        list: A list of nodes representing the computed tour.

    Notes:
        - All nodes are represented as integers.
        - The solution must use dynamic programming.
        - The tour must begin and end at node 0.
        - The tour can only traverse existing edges in the graph.
        - The tour must visit every node in G exactly once.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n <= 1:
        return nodes
    
    # Create node mapping (ensure 0 is the starting node)
    if 0 not in nodes:
        return []
    
    node_to_idx = {node: i for i, node in enumerate(sorted(nodes))}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    start_idx = node_to_idx[0]
    
    # DP table: dp[mask][i] = minimum cost to visit nodes in mask ending at node i
    dp = {}
    parent = {}
    
    # Initialize: starting from node 0, only node 0 visited
    start_mask = 1 << start_idx
    dp[(start_mask, start_idx)] = 0
    
    # Fill DP table for all subset sizes
    for size in range(2, n + 1):
        for subset in combinations(range(n), size):
            if start_idx not in subset:
                continue
                
            mask = sum(1 << i for i in subset)
            
            for curr in subset:
                if curr == start_idx:
                    continue
                    
                prev_mask = mask ^ (1 << curr)
                min_cost = float('inf')
                best_prev = None
                
                for prev in subset:
                    if prev == curr:
                        continue
                    if (prev_mask, prev) not in dp:
                        continue
                        
                    curr_node = idx_to_node[curr]
                    prev_node = idx_to_node[prev]
                    
                    if G.has_edge(prev_node, curr_node):
                        cost = dp[(prev_mask, prev)] + G[prev_node][curr_node].get('weight', 1)
                        if cost < min_cost:
                            min_cost = cost
                            best_prev = prev
                
                if best_prev is not None:
                    dp[(mask, curr)] = min_cost
                    parent[(mask, curr)] = best_prev
    
    # Find minimum cost to complete the tour
    full_mask = (1 << n) - 1
    min_tour_cost = float('inf')
    last_node = None
    
    for i in range(n):
        if i == start_idx:
            continue
        if (full_mask, i) not in dp:
            continue
            
        curr_node = idx_to_node[i]
        start_node = idx_to_node[start_idx]
        
        if G.has_edge(curr_node, start_node):
            cost = dp[(full_mask, i)] + G[curr_node][start_node].get('weight', 1)
            if cost < min_tour_cost:
                min_tour_cost = cost
                last_node = i
    
    if last_node is None:
        return []
    
    # Reconstruct tour
    tour = []
    mask = full_mask
    curr = last_node
    
    while (mask, curr) in parent:
        tour.append(idx_to_node[curr])
        prev = parent[(mask, curr)]
        mask ^= (1 << curr)
        curr = prev
    
    tour.append(idx_to_node[start_idx])  # Add starting node
    tour.reverse()
    tour.append(0)  # Return to start
    
    return tour