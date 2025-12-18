import networkx as nx
import numpy as np
import random
import math
import pulp as pl
from student_utils import analyze_solution


def ptp_solver_greedy(G: nx.DiGraph, H: list, alpha: float):
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

    key_nodes = [0, 0]  # Key nodes in the tour (will be expanded to full path later)
    pick_up_locs_dict = {}
    node_num = G.number_of_nodes()
    dist_matrix: np.ndarray = nx.floyd_warshall_numpy(
        G
    )  # dist_matrix[i][j] is the shortest path distance between node i and node j
    all_shortest_paths = dict(
        nx.all_pairs_dijkstra_path(G)
    )  # Precompute all shortest paths

    def cost(key_nodes: list):
        """Calculate cost based on key nodes (using shortest paths between them)"""
        walking_cost = 0
        for friend in H:
            walking_cost += min(
                dist_matrix[friend][key_nodes[i]] for i in range(len(key_nodes))
            )
        return (
            alpha
            * sum(
                [
                    dist_matrix[key_nodes[i]][key_nodes[i + 1]]
                    for i in range(len(key_nodes) - 1)
                ]
            )
            + walking_cost
        )

    def infeasiblility(key_nodes: list):
        """Check how many friends cannot be picked up"""
        visited = 0
        for friend in H:
            can_be_visited = False
            for neighbor in G.neighbors(friend):
                if neighbor in key_nodes:
                    can_be_visited = True
                    break
            if friend in key_nodes:
                can_be_visited = True
            if can_be_visited:
                visited += 1
        return len(H) - visited

    def least_cost_insert(key_nodes, i) -> list:
        """
        Insert node i into the key_nodes to form a candidate tour.
        """
        min_cost = float("inf")
        min_tour = key_nodes.copy()
        for j in range(len(key_nodes) - 1):
            candidate_tour = key_nodes.copy()
            candidate_tour.insert(j + 1, i)
            c = cost(candidate_tour)
            if c < min_cost:
                min_cost = c
                min_tour = candidate_tour
        return min_tour

    def remove_node(key_nodes, i) -> list:
        """
        Remove node i from the key_nodes to form a candidate tour.
        """
        new_tour = key_nodes.copy()
        if i in new_tour:
            new_tour.remove(i)
        return new_tour

    def expand_tour(key_nodes: list) -> list:
        """
        Expand key nodes to a full tour with all intermediate nodes.
        Uses shortest paths between consecutive key nodes.
        """
        full_tour = []
        for i in range(len(key_nodes) - 1):
            path = all_shortest_paths[key_nodes[i]][key_nodes[i + 1]]
            # Add all nodes except the last one (to avoid duplicates)
            full_tour.extend(path[:-1])
        # Add the final node
        full_tour.append(key_nodes[-1])
        return full_tour

    def get_pick_up_locs_dict(key_nodes) -> dict:
        """
        Get the pick-up locations dictionary from the key nodes.
        Find the shortest pick-up location for each friend.
        """
        pick_up_locs_dict = {}
        for friend in H:
            # Check if friend's home is in key_nodes
            if friend in key_nodes:
                if friend not in pick_up_locs_dict:
                    pick_up_locs_dict[friend] = []
                pick_up_locs_dict[friend].append(friend)
            else:
                # Find the neighbor in key_nodes with shortest distance
                min_dist = float("inf")
                best_pickup = None
                for neighbor in G.neighbors(friend):
                    if neighbor in key_nodes:
                        if dist_matrix[friend][neighbor] < min_dist:
                            min_dist = dist_matrix[friend][neighbor]
                            best_pickup = neighbor

                if best_pickup is not None:
                    if best_pickup not in pick_up_locs_dict:
                        pick_up_locs_dict[best_pickup] = []
                    pick_up_locs_dict[best_pickup].append(friend)
        return pick_up_locs_dict

    while True:
        candidate_tours = []
        for i in range(node_num):
            if i == 0:
                continue  # 0 is the start and end node, so it cannot be removed or inserted
            candidate_tour = (
                remove_node(key_nodes, i)
                if i in key_nodes
                else least_cost_insert(key_nodes, i)
            )
            candidate_tours.append(candidate_tour)

        current_infeasibility = infeasiblility(key_nodes)

        if current_infeasibility == 0:
            feasible_tours = [
                candidate_tour
                for candidate_tour in candidate_tours
                if infeasiblility(candidate_tour) == 0
            ]
            if not feasible_tours:
                break
            new_tour = min(feasible_tours, key=cost)
        else:
            better_tours = [
                candidate_tour
                for candidate_tour in candidate_tours
                if infeasiblility(candidate_tour) < current_infeasibility
            ]
            if not better_tours:
                break
            new_tour = min(better_tours, key=cost)

        if cost(key_nodes) <= cost(new_tour) and current_infeasibility == 0:
            # No cost improvement for feasible tour
            break
        else:
            key_nodes = new_tour

    # Expand key nodes to full tour with all intermediate nodes
    tour = expand_tour(key_nodes)

    # Generate final pick-up locations dictionary
    pick_up_locs_dict = get_pick_up_locs_dict(key_nodes)

    # print(f"[GREEDY] friends: {H}")
    # print(f"[GREEDY] key_nodes: {key_nodes}")
    # print(f"[GREEDY] tour: {tour}")
    # print(f"[GREEDY] pick_up_locs_dict: {pick_up_locs_dict}")
    # print(f"[GREEDY] cost: {cost(tour)}")

    return tour, pick_up_locs_dict


def ptp_solver_multi_start(G: nx.DiGraph, H: list, alpha: float, num_starts: int = 10):
    """
    PTP solver using multi-start strategy.

    Parameters:
        G (nx.DiGraph): A NetworkX graph representing the city.
        H (list): A list of home nodes.
        alpha (float): The coefficient for calculating driving cost.
        num_starts (int): The number of starting points.

    Returns:
        tuple: A tuple containing:
            - tour (list): A list of nodes traversed by your car.
            - pick_up_locs_dict (dict): A dictionary where:
                - Keys are pick-up locations.
                - Values are lists or tuples containing friends who get picked up
                  at that specific pick-up location. Friends are represented by
                  their home nodes.
    """

    n = G.number_of_nodes()
    dist_matrix = nx.floyd_warshall_numpy(G)
    all_shortest_paths = dict(nx.all_pairs_dijkstra_path(G))

    def cost(key_nodes: list):
        walking_cost = 0
        for friend in H:
            walking_cost += min(dist_matrix[friend][key_nodes[i]] for i in range(len(key_nodes)))
        driving_cost = sum(
            [
                dist_matrix[key_nodes[i]][key_nodes[i + 1]]
                for i in range(len(key_nodes) - 1)
            ]
        )
        return alpha * driving_cost + walking_cost

    def is_feasible(key_nodes: list):
        for friend in H:
            can_pickup = False
            if friend in key_nodes:
                can_pickup = True
            else:
                for neighbor in G.neighbors(friend):
                    if neighbor in key_nodes:
                        can_pickup = True
                        break
            if not can_pickup:
                return False
        return True

    def expand_tour(key_nodes: list):
        full_tour = []
        for i in range(len(key_nodes) - 1):
            path = all_shortest_paths[key_nodes[i]][key_nodes[i + 1]]
            full_tour.extend(path[:-1])
        full_tour.append(key_nodes[-1])
        return full_tour

    def get_pick_up_locs_dict(key_nodes):
        """获取接送位置字典"""
        pick_up_locs_dict = {}
        for friend in H:
            if friend in key_nodes:
                if friend not in pick_up_locs_dict:
                    pick_up_locs_dict[friend] = []
                pick_up_locs_dict[friend].append(friend)
            else:
                min_dist = float("inf")
                best_pickup = None
                for neighbor in G.neighbors(friend):
                    if neighbor in key_nodes:
                        if dist_matrix[friend][neighbor] < min_dist:
                            min_dist = dist_matrix[friend][neighbor]
                            best_pickup = neighbor
                if best_pickup is not None:
                    if best_pickup not in pick_up_locs_dict:
                        pick_up_locs_dict[best_pickup] = []
                    pick_up_locs_dict[best_pickup].append(friend)
        return pick_up_locs_dict

    def infeasibility(kn):
        visited = 0
        for friend in H:
            can_be_visited = False
            if friend in kn:
                can_be_visited = True
            else:
                for neighbor in G.neighbors(friend):
                    if neighbor in kn:
                        can_be_visited = True
                        break
            if can_be_visited:
                visited += 1
        return len(H) - visited

    def least_cost_insert(kn, i):
        min_cost = float("inf")
        min_tour = kn.copy()
        for j in range(len(kn) - 1):
            candidate = kn.copy()
            candidate.insert(j + 1, i)
            c = cost(candidate)
            if c < min_cost:
                min_cost = c
                min_tour = candidate
        return min_tour

    def remove_node(kn, i):
        new_tour = kn.copy()
        if i in new_tour:
            new_tour.remove(i)
        return new_tour

    def local_search(initial_key_nodes):
        key_nodes = initial_key_nodes.copy()

        while True:
            candidate_tours = []
            for i in range(n):
                if i == 0:
                    continue
                candidate = (
                    remove_node(key_nodes, i)
                    if i in key_nodes
                    else least_cost_insert(key_nodes, i)
                )
                candidate_tours.append(candidate)

            current_infeasibility = infeasibility(key_nodes)

            if current_infeasibility == 0:
                feasible_tours = [
                    ct for ct in candidate_tours if infeasibility(ct) == 0
                ]
                if not feasible_tours:
                    break
                new_tour = min(feasible_tours, key=cost)
            else:
                better_tours = [
                    ct
                    for ct in candidate_tours
                    if infeasibility(ct) < current_infeasibility
                ]
                if not better_tours:
                    break
                new_tour = min(better_tours, key=cost)

            if cost(key_nodes) <= cost(new_tour) and current_infeasibility == 0:
                break
            else:
                key_nodes = new_tour

        return key_nodes

    def generate_initial_solutions(num_starts):
        initial_solutions = []

        initial_solutions.append([0, 0])

        # random sample friends
        for _ in range(min(3, num_starts // 3)):
            solution = [0]
            friends_sample = random.sample(
                H, min(len(H), random.randint(1, min(5, len(H))))
            )
            solution.extend(friends_sample)
            solution.append(0)
            initial_solutions.append(solution)

        # nearest neighbor
        if len(initial_solutions) < num_starts:
            solution = [0]
            current = 0
            remaining = set(H)
            while remaining:
                nearest = min(remaining, key=lambda h: dist_matrix[current][h])
                solution.append(nearest)
                remaining.remove(nearest)
                current = nearest
            solution.append(0)
            initial_solutions.append(solution)

        # farthest insertion
        if len(initial_solutions) < num_starts:
            solution = [0, 0]
            remaining = set(H)
            while remaining:
                # 找到距离tour最远的点
                farthest = max(
                    remaining,
                    key=lambda h: min(
                        dist_matrix[h][solution[i]] for i in range(len(solution))
                    ),
                )
                # 找到最佳插入位置
                best_pos = 1
                best_cost_increase = float("inf")
                for pos in range(1, len(solution)):
                    cost_increase = (
                        dist_matrix[solution[pos - 1]][farthest]
                        + dist_matrix[farthest][solution[pos]]
                        - dist_matrix[solution[pos - 1]][solution[pos]]
                    )
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_pos = pos
                solution.insert(best_pos, farthest)
                remaining.remove(farthest)
            initial_solutions.append(solution)

        # random generate remaining initial solutions
        while len(initial_solutions) < num_starts:
            solution = [0]
            # 随机选择一些节点
            num_nodes = random.randint(max(1, len(H) // 2), min(n - 1, len(H) + 3))
            candidates = list(range(1, n))
            selected = random.sample(candidates, min(num_nodes, len(candidates)))
            solution.extend(selected)
            solution.append(0)
            initial_solutions.append(solution)

        return initial_solutions

    # 主循环：多起始点搜索
    #print(f"[MULTI-START] Running with {num_starts} different starting points...")

    initial_solutions = generate_initial_solutions(num_starts)
    best_key_nodes = None
    best_cost_value = float("inf")

    for i, initial_solution in enumerate(initial_solutions):
        # 从每个初始解开始局部搜索
        result_key_nodes = local_search(initial_solution)

        # 检查是否可行
        if is_feasible(result_key_nodes):
            result_cost = cost(result_key_nodes)

            if result_cost < best_cost_value:
                best_cost_value = result_cost
                best_key_nodes = result_key_nodes
                #print(f"[MULTI-START] New best at start {i + 1}/{num_starts}: cost={result_cost:.2f}")

    #print(f"[MULTI-START] Final best cost: {best_cost_value:.2f}")

    tour = expand_tour(best_key_nodes)
    pick_up_locs_dict = get_pick_up_locs_dict(best_key_nodes)

    return tour, pick_up_locs_dict


def ptp_solver_ILP(G: nx.DiGraph, H: list, alpha: float):
    """
    PTP solver using Integer Linear Programming (ILP).
    Based on Miller-Tucker-Zemlin (MTZ) formulation

    Parameters:
        G (nx.DiGraph): A NetworkX graph representing the city.
        H (list): A list of home nodes.
        alpha (float): The coefficient for calculating driving cost.

    Returns:
        tuple: (tour, pick_up_locs_dict)
    """

    n = G.number_of_nodes()
    nodes = list(range(n))

    dist_matrix = nx.floyd_warshall_numpy(G)
    all_shortest_paths = dict(nx.all_pairs_dijkstra_path(G))

    prob = pl.LpProblem("PTP_Problem", pl.LpMinimize)

    # 决策变量
    # x[i][j] = 1 表示从节点i到节点j
    x = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                x[i, j] = pl.LpVariable(f"x_{i}_{j}", cat="Binary")

    # u[i]: MTZ约束的辅助变量，表示节点i在tour中的访问顺序
    u = {}
    for i in nodes:
        if i != 0:  # 节点0是起点，不需要u变量
            u[i] = pl.LpVariable(f"u_{i}", lowBound=1, upBound=n - 1, cat="Integer")

    # y[h][p] = 1 表示朋友h在位置p被接送
    # p可以是h的家或者h的邻居
    y = {}
    for h in H:
        # 朋友可以在自己家被接送
        pickup_locations = [h]
        # 或者在邻居节点被接送
        for neighbor in G.neighbors(h):
            pickup_locations.append(neighbor)

        for p in pickup_locations:
            y[h, p] = pl.LpVariable(f"y_{h}_{p}", cat="Binary")

    # 目标函数：最小化 alpha * 驾驶成本 + 步行成本
    driving_cost = pl.lpSum(
        [alpha * dist_matrix[i][j] * x[i, j] for i in nodes for j in nodes if i != j]
    )

    walking_cost = pl.lpSum(
        [dist_matrix[h][p] * y[h, p] for h in H for p in ([h] + list(G.neighbors(h)))]
    )

    prob += driving_cost + walking_cost, "Total_Cost"

    # 约束条件

    # 1. 流守恒约束：从节点0出发必须回到节点0
    prob += pl.lpSum([x[0, j] for j in nodes if j != 0]) == 1, "Start_from_0"
    prob += pl.lpSum([x[i, 0] for i in nodes if i != 0]) == 1, "Return_to_0"

    # 2. 对于每个非0节点，如果它在tour中，入度=出度≤1
    for j in nodes:
        if j != 0:
            prob += (
                (
                    pl.lpSum([x[i, j] for i in nodes if i != j])
                    == pl.lpSum([x[j, k] for k in nodes if k != j])
                ),
                f"Flow_conservation_{j}",
            )
            prob += (
                pl.lpSum([x[i, j] for i in nodes if i != j]) <= 1,
                f"Max_in_degree_{j}",
            )

    # 3. MTZ约束：消除子环
    # u_i - u_j + n*x[i,j] <= n-1 for all i,j != 0, i != j
    for i in nodes:
        if i == 0:
            continue
        for j in nodes:
            if j == 0 or i == j:
                continue
            prob += u[i] - u[j] + n * x[i, j] <= n - 1, f"MTZ_{i}_{j}"

    # 4. 每个朋友必须被接送恰好一次
    for h in H:
        pickup_locations = [h] + list(G.neighbors(h))
        prob += pl.lpSum([y[h, p] for p in pickup_locations]) == 1, f"Pickup_friend_{h}"

    # 5. 接送位置必须在tour中（如果朋友在p被接送，则tour必须经过p）
    for h in H:
        pickup_locations = [h] + list(G.neighbors(h))
        for p in pickup_locations:
            # 如果y[h,p]=1，则节点p必须在tour中
            # 节点p在tour中意味着：sum(x[i,p] for i!=p) >= y[h,p]
            if p == 0:
                # 节点0总是在tour中
                continue
            else:
                prob += (
                    pl.lpSum([x[i, p] for i in nodes if i != p]) >= y[h, p],
                    f"Pickup_location_in_tour_{h}_{p}",
                )

    # 求解
    solver = pl.PULP_CBC_CMD(msg=0)  # 使用CBC求解器，不显示详细信息
    prob.solve(solver)

    # 检查求解状态
    if prob.status != pl.LpStatusOptimal:
        print(f"[WARNING] ILP solver status: {pl.LpStatus[prob.status]}")
        # 如果ILP求解失败，返回简单的贪心解
        return ptp_solver(G, H, alpha)

    # 提取解
    # 构建tour
    tour_edges = []
    for i in nodes:
        for j in nodes:
            if i != j and pl.value(x[i, j]) > 0.5:  # 二元变量应该是0或1
                tour_edges.append((i, j))

    # 从节点0开始构建tour路径
    tour = [0]
    current = 0
    visited = {0}

    while len(visited) < len(tour_edges) + 1:
        # 找到从current出发的下一个节点
        next_node = None
        for i, j in tour_edges:
            if i == current:
                next_node = j
                break

        if next_node is None:
            break

        tour.append(next_node)
        visited.add(next_node)
        current = next_node

        if current == 0:  # 回到起点
            break

    # 扩展tour：在关键节点之间插入最短路径
    full_tour = []
    for i in range(len(tour) - 1):
        path = all_shortest_paths[tour[i]][tour[i + 1]]
        full_tour.extend(path[:-1])
    full_tour.append(tour[-1])

    # 构建pick_up_locs_dict
    pick_up_locs_dict = {}
    for h in H:
        pickup_locations = [h] + list(G.neighbors(h))
        for p in pickup_locations:
            if pl.value(y[h, p]) > 0.5:  # 朋友h在位置p被接送
                if p not in pick_up_locs_dict:
                    pick_up_locs_dict[p] = []
                pick_up_locs_dict[p].append(h)
                break

    # Debug信息
    #print(f"[ILP] Optimal cost: {pl.value(prob.objective):.2f}")
    #print(f"[ILP] Tour: {full_tour}")
    #print(f"[ILP] Pick-up locations: {pick_up_locs_dict}")

    return full_tour, pick_up_locs_dict


def ptp_solver_SA(G: nx.DiGraph, H: list, alpha: float):
    """
    PTP solver using Simulated Annealing.

    Parameters:
        G (nx.DiGraph): A NetworkX graph representing the city.
        H (list): A list of home nodes.
        alpha (float): The coefficient for calculating driving cost.

    Returns:
        tuple: (tour, pick_up_locs_dict)
    """

    n = G.number_of_nodes()
    dist_matrix = nx.floyd_warshall_numpy(G)
    all_shortest_paths = dict(nx.all_pairs_dijkstra_path(G))

    def cost(key_nodes: list):
        walking_cost = 0
        for friend in H:
            walking_cost += min(
                dist_matrix[friend][key_nodes[i]] for i in range(len(key_nodes))
            )
        driving_cost = sum(
            [
                dist_matrix[key_nodes[i]][key_nodes[i + 1]]
                for i in range(len(key_nodes) - 1)
            ]
        )
        return alpha * driving_cost + walking_cost

    def is_feasible(key_nodes: list):
        for friend in H:
            can_pickup = False
            if friend in key_nodes:
                can_pickup = True
            else:
                for neighbor in G.neighbors(friend):
                    if neighbor in key_nodes:
                        can_pickup = True
                        break
            if not can_pickup:
                return False
        return True

    def expand_tour(key_nodes: list):
        full_tour = []
        for i in range(len(key_nodes) - 1):
            path = all_shortest_paths[key_nodes[i]][key_nodes[i + 1]]
            full_tour.extend(path[:-1])
        full_tour.append(key_nodes[-1])
        return full_tour

    def get_pick_up_locs_dict(key_nodes):
        pick_up_locs_dict = {}
        for friend in H:
            if friend in key_nodes:
                if friend not in pick_up_locs_dict:
                    pick_up_locs_dict[friend] = []
                pick_up_locs_dict[friend].append(friend)
            else:
                min_dist = float("inf")
                best_pickup = None
                for neighbor in G.neighbors(friend):
                    if neighbor in key_nodes:
                        if dist_matrix[friend][neighbor] < min_dist:
                            min_dist = dist_matrix[friend][neighbor]
                            best_pickup = neighbor
                if best_pickup is not None:
                    if best_pickup not in pick_up_locs_dict:
                        pick_up_locs_dict[best_pickup] = []
                    pick_up_locs_dict[best_pickup].append(friend)
        return pick_up_locs_dict

    def get_neighbor(current_solution):
        """
        生成邻域解：随机选择一种操作
        1. 插入一个新节点（40%）
        2. 删除一个节点（30%）
        3. 交换两个节点位置（20%）
        4. 2-opt交换（10%）
        """
        new_solution = current_solution.copy()
        operation = random.random()

        if operation < 0.4:  # 插入操作
            # 随机选择一个不在tour中的节点插入
            not_in_tour = [i for i in range(n) if i not in new_solution and i != 0]
            if not_in_tour:
                node_to_insert = random.choice(not_in_tour)
                # 随机选择插入位置（不包括首尾的0）
                insert_pos = random.randint(1, len(new_solution) - 1)
                new_solution.insert(insert_pos, node_to_insert)

        elif operation < 0.7:  # 删除操作
            # 随机删除一个节点（不包括首尾的0）
            if len(new_solution) > 2:
                removable = [i for i in range(1, len(new_solution) - 1)]
                if removable:
                    pos_to_remove = random.choice(removable)
                    new_solution.pop(pos_to_remove)

        elif operation < 0.9:  # 交换操作
            # 随机交换两个节点（不包括首尾的0）
            if len(new_solution) > 3:
                indices = random.sample(
                    range(1, len(new_solution) - 1), min(2, len(new_solution) - 2)
                )
                if len(indices) == 2:
                    new_solution[indices[0]], new_solution[indices[1]] = (
                        new_solution[indices[1]],
                        new_solution[indices[0]],
                    )

        else:  # 2-opt操作
            # 反转一段路径
            if len(new_solution) > 3:
                i, j = sorted(random.sample(range(1, len(new_solution) - 1), 2))
                new_solution[i : j + 1] = reversed(new_solution[i : j + 1])

        return new_solution

    initial_solution = [0, 0]
    for h in H:
        # 尝试插入朋友节点或其邻居
        best_cost = float("inf")
        best_node = h
        for candidate in [h] + list(G.neighbors(h)):
            test_solution = initial_solution[:-1] + [candidate, 0]
            if is_feasible(test_solution):
                c = cost(test_solution)
                if c < best_cost:
                    best_cost = c
                    best_node = candidate
        if best_node not in initial_solution:
            initial_solution.insert(-1, best_node)

    # simulated annealing
    current_solution = initial_solution
    current_cost = cost(current_solution)

    best_solution = current_solution.copy()
    best_cost = current_cost

    T_initial = 1000.0  # initial temperature
    T_final = 0.1  # final temperature
    alpha_cooling = 0.95  # cooling rate (temperature *= alpha)
    iterations_per_temp = 100  # iterations per temperature

    T = T_initial
    iteration = 0
    max_iterations = 10000

    #print("[SA] Starting simulated annealing...")
    #print(f"[SA] Initial cost: {current_cost:.2f}")

    while T > T_final and iteration < max_iterations:
        for _ in range(iterations_per_temp):
            new_solution = get_neighbor(current_solution)

            if not is_feasible(new_solution):
                continue

            new_cost = cost(new_solution)
            delta_cost = new_cost - current_cost

            if delta_cost < 0:
                accept = True
            else:
                # accept with probability
                probability = math.exp(-delta_cost / T)
                accept = random.random() < probability

            if accept:
                current_solution = new_solution
                current_cost = new_cost

                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
                    # print(f"[SA] New best at T={T:.2f}: {best_cost:.2f}")

            iteration += 1

        # cooling
        T *= alpha_cooling

    #print(f"[SA] Final best cost: {best_cost:.2f}")
    #print(f"[SA] Total iterations: {iteration}")

    tour = expand_tour(best_solution)
    pick_up_locs_dict = get_pick_up_locs_dict(best_solution)

    return tour, pick_up_locs_dict


def ptp_solver(G: nx.DiGraph, H: list, alpha: float):
    """
    PTP solver.
    """
    node_num = G.number_of_nodes()
    use_ILP = node_num <= 20
    repeat_times = 20
    # verbose = True
    verbose = False

    solutions = []

    if use_ILP:
        IP_tour, IP_pick_up_locs_dict = ptp_solver_ILP(G, H, alpha)
        IP_is_legitimate, IP_driving_cost, IP_walking_cost = analyze_solution(G, H, alpha, IP_tour, IP_pick_up_locs_dict)
        solutions.append(("IP", IP_is_legitimate, IP_driving_cost, IP_walking_cost, IP_tour, IP_pick_up_locs_dict))

    GD_tour, GD_pick_up_locs_dict = ptp_solver_greedy(G, H, alpha)
    GD_is_legitimate, GD_driving_cost, GD_walking_cost = analyze_solution(G, H, alpha, GD_tour, GD_pick_up_locs_dict)
    solutions.append(("GD", GD_is_legitimate, GD_driving_cost, GD_walking_cost, GD_tour, GD_pick_up_locs_dict))

    for k in range(repeat_times):
        SA_tour, SA_pick_up_locs_dict = ptp_solver_SA(G, H, alpha)
        SA_is_legitimate, SA_driving_cost, SA_walking_cost = analyze_solution(G, H, alpha, SA_tour, SA_pick_up_locs_dict)
        SA_name = f"SA_{k}" if k > 0 else "SA"
        solutions.append((SA_name, SA_is_legitimate, SA_driving_cost, SA_walking_cost, SA_tour, SA_pick_up_locs_dict))


        MS_tour, MS_pick_up_locs_dict = ptp_solver_multi_start(G, H, alpha)
        MS_is_legitimate, MS_driving_cost, MS_walking_cost = analyze_solution(G, H, alpha, MS_tour, MS_pick_up_locs_dict)
        MS_name = f"MS_{k}" if k > 0 else "MS"
        solutions.append((MS_name, MS_is_legitimate, MS_driving_cost, MS_walking_cost, MS_tour, MS_pick_up_locs_dict))

    solutions.sort(key=lambda x: x[2] + x[3]) # sort by total cost


    if verbose:
        print("=" * 70)
        print(f"{'Method':<6} | {'Legit':<9} | {'Driving Cost':>13} | {'Walking Cost':>13} | {'Total Cost':>11}")
        for solution in solutions:
            print(f"{solution[0]:<6} | {str(solution[1]):<9} | {solution[2]:13.3f} | {solution[3]:13.3f} | {(solution[2] + solution[3]):11.3f}")
        print("-" * 70)
        print(f"[Best method] {solutions[0][0]}")
        print("=" * 70)

    return solutions[0][4], solutions[0][5]

if __name__ == "__main__":
    pass