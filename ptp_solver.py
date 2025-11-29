import networkx as nx
from student_utils import *

def ptp_solver(G:nx.DiGraph, H:list, alpha:float):
    """
    PTP solver.

    Parameters:
        G (nx.DiGraph): A NetworkX graph representing the city.
            This directed graph is equivalent to an undirected one by construction.
        H (list): A list of home nodes.
        alpha (float): The coefficient for calculating cost.

    Returns:
        tuple: A tuple containing:
            - tour (list): A list of nodes traversed by your car.
            - pick_up_locs_dict (dict): A dictionary where:
                - Keys are pick-up locations.
                - Values are lists or tuples containing friends who get picked up
                  at that specific pick-up location. Friends are represented by
                  their home nodes.

    Notes:
    - All nodes are represented as integers.
    - The tour must begin and end at node 0.
    - The tour can only go through existing edges in the graph.
    - Pick-up locations must be part of the tour.
    - Each friend should be picked up exactly once.
    - The pick-up locations must be neighbors of the friends' home nodes or their homes.
    """

    return tour, pick_up_locs_dict


if __name__ == "__main__":
    pass
