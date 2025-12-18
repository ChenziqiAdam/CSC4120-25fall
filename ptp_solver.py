import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pulp as pl

from student_utils import analyze_solution


# -----------------------------
# Shortest path cache utilities
# -----------------------------

@dataclass(frozen=True)
class _EvalMetrics:
    """Cached evaluation results for a key-node tour."""
    full_tour: Tuple[int, ...]
    visited: frozenset
    driving_len: float
    walking_allowed: float
    infeasibility: int
    walking_relaxed: float


class ShortestPathCache:
    """
    Cache all-pairs shortest-path distances + per-source predecessor trees
    to allow fast path reconstruction without storing all paths explicitly.
    Assumes nodes are labeled 0..n-1 (as in the course project).
    """

    def __init__(self, G: nx.DiGraph):
        self.G = G
        self.n = G.number_of_nodes()
        # Use an undirected view for neighborhood (PTP defines neighbors in the underlying city graph)
        self._UG = G.to_undirected(as_view=True)

        # Precompute predecessors and distances using Dijkstra from every source.
        # This is typically much cheaper than Floyd-Warshall on sparse graphs and avoids path explosion.
        # NetworkX function availability differs across versions; we support both APIs.
        self._pred: List[Dict[int, List[int]]] = [dict() for _ in range(self.n)]
        self._dist: np.ndarray = np.full((self.n, self.n), np.inf, dtype=float)

        # Project graphs are labeled 0..n-1; keep a defensive check for clarity.
        nodes = list(G.nodes())
        if any((not isinstance(v, int) or v < 0 or v >= self.n) for v in nodes):
            raise ValueError(
                "This solver expects graph nodes to be integer labels in [0, n-1]. "
                f"Got node labels: {sorted(nodes)[:10]}{'...' if len(nodes) > 10 else ''}"
            )

        try:
            # Newer NetworkX (some versions) expose this all-pairs convenience iterator.
            iterator = nx.all_pairs_dijkstra_predecessor_and_distance(G, weight="weight")
            for src, (pred, dist) in iterator:
                self._pred[src] = pred
                for tgt, d in dist.items():
                    self._dist[src, tgt] = float(d)
        except AttributeError:
            # Compatibility path for older NetworkX: run single-source Dijkstra for each source.
            for src in range(self.n):
                pred, dist = nx.dijkstra_predecessor_and_distance(G, src, weight="weight")
                self._pred[src] = pred
                for tgt, d in dist.items():
                    self._dist[src, tgt] = float(d)

        # Cache reconstructed paths (u, v) -> tuple(path nodes)
        @lru_cache(maxsize=200_000)
        def _path(u: int, v: int) -> Optional[Tuple[int, ...]]:
            if u == v:
                return (u,)
            # Backtrack from v to u using predecessor tree rooted at u
            pred_u = self._pred[u]
            cur = v
            rev = [cur]
            while cur != u:
                plist = pred_u.get(cur)
                if not plist:
                    return None
                cur = plist[0]
                rev.append(cur)
            rev.reverse()
            return tuple(rev)

        self._path = _path

    def neighbors(self, u: int) -> List[int]:
        return list(self._UG.neighbors(u))

    def dist(self, u: int, v: int) -> float:
        return float(self._dist[u, v])

    def path(self, u: int, v: int) -> Optional[List[int]]:
        p = self._path(u, v)
        return list(p) if p is not None else None


# -----------------------------
# PTP objective helpers
# -----------------------------

def _candidate_nodes(G: nx.DiGraph, H: Sequence[int], sp: ShortestPathCache) -> List[int]:
    """
    Restrict the move set to nodes that can matter:
    {0} ∪ H ∪ N(H). Because we drive on metric shortest-path distances,
    inserting arbitrary extra waypoints cannot reduce driving length.
    """
    cand: Set[int] = {0}
    for h in H:
        cand.add(h)
        for nb in sp.neighbors(h):
            cand.add(nb)
    return sorted(cand)


def _expand_key_nodes_to_full_tour(key_nodes: Sequence[int], sp: ShortestPathCache) -> Optional[Tuple[int, ...]]:
    """
    Expand a key-node tour into the explicit node-by-node tour by concatenating
    shortest paths between consecutive key nodes.
    """
    if len(key_nodes) < 2:
        return None
    full: List[int] = []
    for a, b in zip(key_nodes[:-1], key_nodes[1:]):
        p = sp.path(int(a), int(b))
        if p is None:
            return None
        # Append all but last to avoid duplication
        full.extend(p[:-1])
    full.append(int(key_nodes[-1]))
    return tuple(full)


def _walking_cost_allowed(visited: Set[int], H: Sequence[int], sp: ShortestPathCache) -> Tuple[float, int]:
    """
    Walking cost under the true PTP constraint:
    friend h can be picked up only at {h} ∪ N(h), and the pickup node must be visited by the car.
    Returns (walking_cost, infeasibility_count).
    """
    total = 0.0
    infeasible = 0
    for h in H:
        allowed = [h] + sp.neighbors(h)
        candidates = [p for p in allowed if p in visited]
        if not candidates:
            infeasible += 1
            continue
        total += min(sp.dist(h, p) for p in candidates)
    return total, infeasible


def _walking_cost_relaxed(key_nodes: Sequence[int], H: Sequence[int], sp: ShortestPathCache) -> float:
    """
    Relaxed walking cost used only to guide search while infeasible:
    allow walking to any visited *key* node (not full tour) to keep evaluation cheap.
    """
    if not key_nodes:
        return float("inf")
    total = 0.0
    for h in H:
        total += min(sp.dist(h, p) for p in key_nodes)
    return total


def _driving_len(key_nodes: Sequence[int], sp: ShortestPathCache) -> float:
    d = 0.0
    for a, b in zip(key_nodes[:-1], key_nodes[1:]):
        w = sp.dist(a, b)
        if math.isinf(w):
            return float("inf")
        d += w
    return d


def _build_pickup_dict_from_visited(visited: Set[int], H: Sequence[int], sp: ShortestPathCache) -> Dict[int, List[int]]:
    """
    Build a pickup dictionary that satisfies the PTP pickup constraint.
    """
    d: Dict[int, List[int]] = {}
    for h in H:
        allowed = [h] + sp.neighbors(h)
        candidates = [p for p in allowed if p in visited]
        if not candidates:
            # Leave unassigned; caller should ensure feasibility before using this.
            continue
        best = min(candidates, key=lambda p: sp.dist(h, p))
        d.setdefault(best, []).append(h)
    return d


def _fallback_feasible(G: nx.DiGraph, H: Sequence[int], sp: ShortestPathCache) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Always-feasible baseline: visit every home node once (unique), return to 0.
    """
    key_nodes = [0] + sorted(set(int(h) for h in H if int(h) != 0)) + [0]
    full = _expand_key_nodes_to_full_tour(key_nodes, sp)
    if full is None:
        # As a last resort, return a trivial 0->0 tour (may be invalid if H nonempty)
        return [0, 0], {}
    visited = set(full)
    pickup = _build_pickup_dict_from_visited(visited, H, sp)
    return list(full), pickup


# -----------------------------
# Greedy + Local Search Solver
# -----------------------------

def ptp_solver_greedy(G: nx.DiGraph, H: list, alpha: float, sp: Optional[ShortestPathCache] = None):
    """
    Greedy / local-search PTP solver.

    Returns:
        (tour, pick_up_locs_dict)
    """
    if sp is None:
        sp = ShortestPathCache(G)

    n = sp.n
    cand_nodes = _candidate_nodes(G, H, sp)

    # Start from a guaranteed-feasible key-node list: visit all homes (unique).
    key_nodes: List[int] = [0] + sorted(set(int(h) for h in H if int(h) != 0)) + [0]
    if len(key_nodes) < 2:
        key_nodes = [0, 0]

    @lru_cache(maxsize=50_000)
    def _metrics(kn: Tuple[int, ...]) -> _EvalMetrics:
        kn_list = list(kn)
        full = _expand_key_nodes_to_full_tour(kn_list, sp)
        if full is None:
            return _EvalMetrics(full_tour=tuple([0, 0]), visited=frozenset({0}), driving_len=float("inf"),
                                walking_allowed=float("inf"), infeasibility=len(H), walking_relaxed=float("inf"))

        visited = set(full)
        drive = _driving_len(kn_list, sp)
        walk_allowed, infeas = _walking_cost_allowed(visited, H, sp)
        walk_relaxed = _walking_cost_relaxed(kn_list, H, sp)
        return _EvalMetrics(full_tour=full, visited=frozenset(visited), driving_len=drive,
                            walking_allowed=walk_allowed, infeasibility=infeas, walking_relaxed=walk_relaxed)

    def true_total(kn: Sequence[int]) -> float:
        m = _metrics(tuple(kn))
        if m.infeasibility != 0:
            return float("inf")
        return alpha * m.driving_len + m.walking_allowed

    def infeasibility(kn: Sequence[int]) -> int:
        return _metrics(tuple(kn)).infeasibility

    def penalized_total(kn: Sequence[int]) -> float:
        m = _metrics(tuple(kn))
        return alpha * m.driving_len + m.walking_relaxed

    def best_insertion(kn: List[int], node: int) -> List[int]:
        if node in kn:
            return kn
        best = None
        best_val = float("inf")
        for pos in range(1, len(kn)):  # insert before pos
            cand = kn[:pos] + [node] + kn[pos:]
            val = penalized_total(cand)
            if val < best_val:
                best_val = val
                best = cand
        return best if best is not None else kn

    def remove_node(kn: List[int], node: int) -> List[int]:
        if node == 0:
            return kn
        if node not in kn:
            return kn
        # remove a single occurrence (key_nodes rarely have duplicates except 0)
        out = kn.copy()
        out.remove(node)
        if out[0] != 0:
            out.insert(0, 0)
        if out[-1] != 0:
            out.append(0)
        if len(out) < 2:
            out = [0, 0]
        return out

    # Local search: best improvement over insert/remove moves within candidate set
    max_rounds = 200
    for _ in range(max_rounds):
        cur_inf = infeasibility(key_nodes)
        cur_true = true_total(key_nodes)

        best_kn = key_nodes
        best_inf = cur_inf
        best_val = cur_true if cur_inf == 0 else penalized_total(key_nodes)

        for v in cand_nodes:
            if v == 0:
                continue
            if v in key_nodes:
                cand = remove_node(key_nodes, v)
            else:
                cand = best_insertion(key_nodes, v)

            c_inf = infeasibility(cand)
            if c_inf < best_inf:
                best_inf = c_inf
                best_kn = cand
                best_val = true_total(cand) if c_inf == 0 else penalized_total(cand)
            elif c_inf == best_inf:
                c_val = true_total(cand) if c_inf == 0 else penalized_total(cand)
                if c_val < best_val - 1e-9:
                    best_kn = cand
                    best_val = c_val

        # Stopping: feasible and no improvement
        if cur_inf == 0 and best_inf == 0 and best_val >= cur_true - 1e-9:
            break

        # Otherwise take the best move found
        if best_kn == key_nodes:
            # No change possible -> stop
            break
        key_nodes = best_kn

    m = _metrics(tuple(key_nodes))
    tour = list(m.full_tour)
    pickup = _build_pickup_dict_from_visited(set(m.visited), H, sp)

    if m.infeasibility != 0:
        return _fallback_feasible(G, H, sp)

    return tour, pickup


# -----------------------------
# Multi-start wrapper
# -----------------------------

def ptp_solver_multi_start(G: nx.DiGraph, H: list, alpha: float, num_starts: int = 10, sp: Optional[ShortestPathCache] = None):
    """
    Multi-start strategy: generate multiple initial key-node tours and run greedy local search.
    """
    if sp is None:
        sp = ShortestPathCache(G)

    cand_nodes = _candidate_nodes(G, H, sp)
    n = sp.n

    def random_initial() -> List[int]:
        # Start from a subset of candidate pickup nodes; always keep 0 at ends.
        base = [v for v in cand_nodes if v != 0]
        if not base:
            return [0, 0]
        k = random.randint(max(1, len(H) // 2), min(len(base), len(H) + 3))
        chosen = random.sample(base, k)
        return [0] + chosen + [0]

    initial_solutions: List[List[int]] = []
    # Deterministic baseline
    initial_solutions.append([0] + sorted(set(int(h) for h in H if int(h) != 0)) + [0])
    # Add random starts
    while len(initial_solutions) < max(1, num_starts):
        initial_solutions.append(random_initial())

    best_tour = None
    best_pickup = None
    best_cost = float("inf")

    for init in initial_solutions:
        tour, pickup = ptp_solver_greedy(G, H, alpha, sp=sp) if init == initial_solutions[0] else _run_greedy_from_init(G, H, alpha, init, sp)
        legit, drive, walk = analyze_solution(G, H, alpha, tour, pickup)
        if legit:
            total = drive + walk
            if total < best_cost:
                best_cost = total
                best_tour = tour
                best_pickup = pickup

    if best_tour is None:
        return _fallback_feasible(G, H, sp)
    return best_tour, best_pickup


def _run_greedy_from_init(G: nx.DiGraph, H: list, alpha: float, init_key_nodes: List[int], sp: ShortestPathCache):
    """
    Run the greedy local-search routine starting from a provided initial key-node tour.
    """
    # Slightly hacky reuse: temporarily seed the greedy by replacing its start state.
    # We implement a minimal local-search here mirroring ptp_solver_greedy but with a given init.
    cand_nodes = _candidate_nodes(G, H, sp)
    key_nodes = init_key_nodes.copy()
    if not key_nodes or key_nodes[0] != 0:
        key_nodes = [0] + [v for v in key_nodes if v != 0] + [0]
    if key_nodes[-1] != 0:
        key_nodes.append(0)
    if len(key_nodes) < 2:
        key_nodes = [0, 0]

    @lru_cache(maxsize=30_000)
    def _metrics(kn: Tuple[int, ...]) -> _EvalMetrics:
        kn_list = list(kn)
        full = _expand_key_nodes_to_full_tour(kn_list, sp)
        if full is None:
            return _EvalMetrics(full_tour=tuple([0, 0]), visited=frozenset({0}), driving_len=float("inf"),
                                walking_allowed=float("inf"), infeasibility=len(H), walking_relaxed=float("inf"))

        visited = set(full)
        drive = _driving_len(kn_list, sp)
        walk_allowed, infeas = _walking_cost_allowed(visited, H, sp)
        walk_relaxed = _walking_cost_relaxed(kn_list, H, sp)
        return _EvalMetrics(full_tour=full, visited=frozenset(visited), driving_len=drive,
                            walking_allowed=walk_allowed, infeasibility=infeas, walking_relaxed=walk_relaxed)

    def true_total(kn: Sequence[int]) -> float:
        m = _metrics(tuple(kn))
        if m.infeasibility != 0:
            return float("inf")
        return alpha * m.driving_len + m.walking_allowed

    def infeasibility(kn: Sequence[int]) -> int:
        return _metrics(tuple(kn)).infeasibility

    def penalized_total(kn: Sequence[int]) -> float:
        m = _metrics(tuple(kn))
        return alpha * m.driving_len + m.walking_relaxed

    def best_insertion(kn: List[int], node: int) -> List[int]:
        if node in kn:
            return kn
        best = None
        best_val = float("inf")
        for pos in range(1, len(kn)):
            cand = kn[:pos] + [node] + kn[pos:]
            val = penalized_total(cand)
            if val < best_val:
                best_val = val
                best = cand
        return best if best is not None else kn

    def remove_node(kn: List[int], node: int) -> List[int]:
        if node == 0 or node not in kn:
            return kn
        out = kn.copy()
        out.remove(node)
        if out[0] != 0:
            out.insert(0, 0)
        if out[-1] != 0:
            out.append(0)
        if len(out) < 2:
            out = [0, 0]
        return out

    max_rounds = 150
    for _ in range(max_rounds):
        cur_inf = infeasibility(key_nodes)
        cur_true = true_total(key_nodes)

        best_kn = key_nodes
        best_inf = cur_inf
        best_val = cur_true if cur_inf == 0 else penalized_total(key_nodes)

        for v in cand_nodes:
            if v == 0:
                continue
            cand = remove_node(key_nodes, v) if v in key_nodes else best_insertion(key_nodes, v)
            c_inf = infeasibility(cand)
            if c_inf < best_inf:
                best_inf = c_inf
                best_kn = cand
                best_val = true_total(cand) if c_inf == 0 else penalized_total(cand)
            elif c_inf == best_inf:
                c_val = true_total(cand) if c_inf == 0 else penalized_total(cand)
                if c_val < best_val - 1e-9:
                    best_kn = cand
                    best_val = c_val

        if cur_inf == 0 and best_inf == 0 and best_val >= cur_true - 1e-9:
            break
        if best_kn == key_nodes:
            break
        key_nodes = best_kn

    m = _metrics(tuple(key_nodes))
    if m.infeasibility != 0:
        return _fallback_feasible(G, H, sp)
    tour = list(m.full_tour)
    pickup = _build_pickup_dict_from_visited(set(m.visited), H, sp)
    return tour, pickup


# -----------------------------
# ILP solver (small graphs)
# -----------------------------

def ptp_solver_ILP(G: nx.DiGraph, H: list, alpha: float, time_limit: int = 30, sp: Optional[ShortestPathCache] = None):
    """
    Exact ILP using MTZ formulation (intended for small graphs).
    Falls back to heuristic if not solved to optimality.
    """
    if sp is None:
        sp = ShortestPathCache(G)

    n = sp.n
    nodes = list(range(n))

    prob = pl.LpProblem("PTP_Problem", pl.LpMinimize)

    # Decision variables
    x = {(i, j): pl.LpVariable(f"x_{i}_{j}", cat="Binary") for i in nodes for j in nodes if i != j}

    # Node visitation indicator: z[i] = 1 iff node i is visited in the tour (i != 0)
    z = {i: pl.LpVariable(f"z_{i}", cat="Binary") for i in nodes if i != 0}

    # MTZ ordering variables
    u = {i: pl.LpVariable(f"u_{i}", lowBound=1, upBound=n - 1, cat="Integer") for i in nodes if i != 0}

    # Pickup assignment y[h,p] for allowed pickup nodes p in {h} ∪ N(h)
    y = {}
    for h in H:
        for p in [h] + sp.neighbors(h):
            y[(h, p)] = pl.LpVariable(f"y_{h}_{p}", cat="Binary")

    # Objective components
    driving_cost = alpha * pl.lpSum([sp.dist(i, j) * x[(i, j)] for (i, j) in x.keys()])
    walking_cost = pl.lpSum([sp.dist(h, p) * y[(h, p)] for (h, p) in y.keys()])
    prob += driving_cost + walking_cost

    # Degree constraints for node 0
    prob += pl.lpSum([x[(0, j)] for j in nodes if j != 0]) == 1
    prob += pl.lpSum([x[(i, 0)] for i in nodes if i != 0]) == 1

    # Flow constraints + define z
    for j in nodes:
        if j == 0:
            continue
        prob += pl.lpSum([x[(i, j)] for i in nodes if i != j]) == z[j]
        prob += pl.lpSum([x[(j, k)] for k in nodes if k != j]) == z[j]

    # MTZ subtour elimination: only active when both nodes are visited
    for i in nodes:
        if i == 0:
            continue
        for j in nodes:
            if j == 0 or i == j:
                continue
            prob += u[i] - u[j] + (n - 1) * x[(i, j)] <= (n - 2) + (n - 1) * (2 - z[i] - z[j])

    # Pickup assignment: each friend assigned exactly once among allowed nodes
    for h in H:
        prob += pl.lpSum([y[(h, p)] for p in [h] + sp.neighbors(h)]) == 1

    # If a pickup node p is chosen, it must be visited in the tour (or be 0? p is never 0 typically)
    for (h, p), var in y.items():
        if p == 0:
            continue
        prob += var <= z[p]

    # Solve
    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    status = prob.solve(solver)

    if pl.LpStatus[status] != "Optimal":
        # Robust fallback (avoid recursion)
        return ptp_solver_multi_start(G, H, alpha, num_starts=10, sp=sp)

    # Reconstruct tour edges
    succ = {}
    for (i, j), var in x.items():
        if var.value() is not None and var.value() > 0.5:
            succ[i] = j

    if 0 not in succ:
        return ptp_solver_multi_start(G, H, alpha, num_starts=10, sp=sp)

    key_nodes = [0]
    cur = 0
    seen = set([0])
    while True:
        cur = succ.get(cur, None)
        if cur is None:
            break
        key_nodes.append(cur)
        if cur == 0:
            break
        if cur in seen:
            break
        seen.add(cur)

    if key_nodes[-1] != 0:
        key_nodes.append(0)

    full = _expand_key_nodes_to_full_tour(key_nodes, sp)
    if full is None:
        return ptp_solver_multi_start(G, H, alpha, num_starts=10, sp=sp)

    visited = set(full)

    pickup = {}
    for h in H:
        chosen = None
        for p in [h] + sp.neighbors(h):
            var = y.get((h, p))
            if var is not None and var.value() is not None and var.value() > 0.5:
                chosen = p
                break
        if chosen is None:
            # should not happen in optimal
            continue
        pickup.setdefault(chosen, []).append(h)

    # Ensure feasibility w.r.t. actually visited set; if mismatch, rebuild safely
    walk_allowed, infeas = _walking_cost_allowed(visited, H, sp)
    if infeas != 0:
        pickup = _build_pickup_dict_from_visited(visited, H, sp)

    return list(full), pickup


# -----------------------------
# Simulated Annealing Solver
# -----------------------------

def ptp_solver_SA(G: nx.DiGraph, H: list, alpha: float, sp: Optional[ShortestPathCache] = None):
    """
    Simulated annealing over key-node tours.
    Only accepts feasible solutions during the SA walk, ensuring output feasibility.
    """
    if sp is None:
        sp = ShortestPathCache(G)

    cand_nodes = _candidate_nodes(G, H, sp)
    base_nodes = [v for v in cand_nodes if v != 0]

    # Start from feasible baseline
    init = [0] + sorted(set(int(h) for h in H if int(h) != 0)) + [0]
    if len(init) < 2:
        init = [0, 0]

    @lru_cache(maxsize=50_000)
    def _metrics(kn: Tuple[int, ...]) -> _EvalMetrics:
        kn_list = list(kn)
        full = _expand_key_nodes_to_full_tour(kn_list, sp)
        if full is None:
            return _EvalMetrics(full_tour=tuple([0, 0]), visited=frozenset({0}), driving_len=float("inf"),
                                walking_allowed=float("inf"), infeasibility=len(H), walking_relaxed=float("inf"))
        visited = set(full)
        drive = _driving_len(kn_list, sp)
        walk_allowed, infeas = _walking_cost_allowed(visited, H, sp)
        walk_relaxed = _walking_cost_relaxed(kn_list, H, sp)
        return _EvalMetrics(full_tour=full, visited=frozenset(visited), driving_len=drive,
                            walking_allowed=walk_allowed, infeasibility=infeas, walking_relaxed=walk_relaxed)

    def is_feasible(kn: Sequence[int]) -> bool:
        return _metrics(tuple(kn)).infeasibility == 0

    def total_cost(kn: Sequence[int]) -> float:
        m = _metrics(tuple(kn))
        if m.infeasibility != 0:
            return float("inf")
        return alpha * m.driving_len + m.walking_allowed

    def get_neighbor(kn: List[int]) -> List[int]:
        new = kn.copy()
        if len(new) < 2:
            return [0, 0]
        # Ensure endpoints are 0
        new[0] = 0
        new[-1] = 0

        op = random.random()
        if op < 0.35:  # insertion
            if base_nodes:
                node = random.choice(base_nodes)
                if node not in new:
                    pos = random.randint(1, max(1, len(new) - 1))
                    new.insert(pos, node)

        elif op < 0.70:  # deletion
            removable = [i for i in range(1, len(new) - 1)]
            if removable and len(new) > 2:
                idx = random.choice(removable)
                new.pop(idx)

        elif op < 0.90:  # swap
            if len(new) > 3:
                i, j = random.sample(range(1, len(new) - 1), 2)
                new[i], new[j] = new[j], new[i]

        else:  # 2-opt
            if len(new) > 4:
                i, j = sorted(random.sample(range(1, len(new) - 1), 2))
                new[i:j+1] = reversed(new[i:j+1])

        # Clean up duplicates of 0 inside
        new = [0] + [v for v in new[1:-1] if v != 0] + [0]
        if len(new) < 2:
            new = [0, 0]
        return new

    current = init
    # Ensure starting feasible (it is, by construction)
    if not is_feasible(current):
        tour, pickup = _fallback_feasible(G, H, sp)
        return tour, pickup

    cur_cost = total_cost(current)
    best = current.copy()
    best_cost = cur_cost

    T = 800.0
    T_final = 0.5
    cooling = 0.97
    iters_per_T = 80
    max_outer = 250  # caps runtime

    for _ in range(max_outer):
        for _ in range(iters_per_T):
            cand = get_neighbor(current)
            if not is_feasible(cand):
                continue
            c_cost = total_cost(cand)
            delta = c_cost - cur_cost
            if delta <= 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / max(T, 1e-9))

            if accept:
                current = cand
                cur_cost = c_cost
                if cur_cost < best_cost - 1e-9:
                    best = current.copy()
                    best_cost = cur_cost

        T *= cooling
        if T < T_final:
            break

    m = _metrics(tuple(best))
    tour = list(m.full_tour)
    pickup = _build_pickup_dict_from_visited(set(m.visited), H, sp)
    if m.infeasibility != 0:
        return _fallback_feasible(G, H, sp)
    return tour, pickup


# -----------------------------
# Top-level orchestrator
# -----------------------------


def _summarize_best_costs(solutions: List[Tuple[str, bool, float, float, List[int], Dict[int, List[int]]]]):
    """
    Given a list of (name, legitimate, driving_cost, walking_cost, tour, pickup_dict),
    aggregate the best (minimum total) solution per algorithm family.

    Families:
      - Greedy
      - ILP
      - SA   (min over SA, SA_1, SA_2, ...)
      - MS   (min over MS, MS_1, MS_2, ...)
    """
    families = {"Greedy": [], "ILP": [], "SA": [], "MS": []}
    for name, legit, drive, walk, tour, pick in solutions:
        if name.startswith("SA"):
            families["SA"].append((name, legit, drive, walk))
        elif name.startswith("MS"):
            families["MS"].append((name, legit, drive, walk))
        elif name == "Greedy":
            families["Greedy"].append((name, legit, drive, walk))
        elif name == "ILP":
            families["ILP"].append((name, legit, drive, walk))

    best = {}
    for fam, items in families.items():
        if not items:
            continue
        legit_items = [it for it in items if it[1] is True]
        pool = legit_items if legit_items else items  # fall back to best even if illegal
        # minimize total
        sel = min(pool, key=lambda it: (it[2] + it[3]))
        best[fam] = {
            "variant": sel[0],
            "legitimate": bool(sel[1]),
            "driving_cost": float(sel[2]),
            "walking_cost": float(sel[3]),
            "total_cost": float(sel[2] + sel[3]),
        }
    return best


def _print_cost_report(best_costs: Dict[str, Dict], chosen: Optional[Tuple[str, bool, float, float]] = None):
    """
    Print a compact report of best costs per algorithm.
    """
    # Stable order
    order = ["Greedy", "ILP", "SA", "MS"]
    print("=== PTP Solver Cost Report (driving + walking) ===")
    for fam in order:
        if fam not in best_costs:
            continue
        b = best_costs[fam]
        tag = "OK" if b["legitimate"] else "ILLEGAL"
        print(
            f"{fam:6s} | best={b['total_cost']:.6f} (drive={b['driving_cost']:.6f}, walk={b['walking_cost']:.6f})"
            f" | variant={b['variant']} | {tag}"
        )
    if chosen is not None:
        n, legit, drive, walk = chosen
        print(
            f"Chosen | {n} | total={drive+walk:.6f} (drive={drive:.6f}, walk={walk:.6f}) | "
            + ("OK" if legit else "ILLEGAL")
        )
    print("===============================================")


def ptp_solver(G: nx.DiGraph, H: list, alpha: float, report: bool = False, return_report: bool = False):
    """
    Main entry point. Tries:
      - ILP for very small graphs,
      - SA + Multi-start for robustness,
      - Greedy as baseline,
    then selects the best *legitimate* solution.
    """
    sp = ShortestPathCache(G)
    node_num = sp.n

    solutions = []

    # Baselines
    g_tour, g_pick = ptp_solver_greedy(G, H, alpha, sp=sp)
    legit, drive, walk = analyze_solution(G, H, alpha, g_tour, g_pick)
    solutions.append(("Greedy", legit, drive, walk, g_tour, g_pick))

    # ILP on small graphs
    if node_num <= 20:
        ilp_tour, ilp_pick = ptp_solver_ILP(G, H, alpha, time_limit=30, sp=sp)
        legit, drive, walk = analyze_solution(G, H, alpha, ilp_tour, ilp_pick)
        solutions.append(("ILP", legit, drive, walk, ilp_tour, ilp_pick))

    # Heuristic repeats (reduced adaptively for larger graphs)
    repeats = 12 if node_num <= 150 else 6
    for k in range(repeats):
        sa_tour, sa_pick = ptp_solver_SA(G, H, alpha, sp=sp)
        legit, drive, walk = analyze_solution(G, H, alpha, sa_tour, sa_pick)
        name = f"SA_{k}" if k > 0 else "SA"
        solutions.append((name, legit, drive, walk, sa_tour, sa_pick))

        ms_tour, ms_pick = ptp_solver_multi_start(G, H, alpha, num_starts=8, sp=sp)
        legit, drive, walk = analyze_solution(G, H, alpha, ms_tour, ms_pick)
        name = f"MS_{k}" if k > 0 else "MS"
        solutions.append((name, legit, drive, walk, ms_tour, ms_pick))

    # Keep only legitimate; otherwise fall back to guaranteed-feasible
    legit_solutions = [s for s in solutions if s[1] is True]
    if not legit_solutions:
        tour, pickup = _fallback_feasible(G, H, sp)
        # fallback is constructed to be legitimate, but we still use the official evaluator for costs
        fb_legit, fb_drive, fb_walk = analyze_solution(G, H, alpha, tour, pickup)
        if report or return_report:
            best_costs = _summarize_best_costs(solutions)
            if report:
                _print_cost_report(best_costs, chosen=("Fallback", fb_legit, fb_drive, fb_walk))
            if return_report:
                return tour, pickup, best_costs
        return tour, pickup

    legit_solutions.sort(key=lambda x: x[2] + x[3])  # total cost
    best = legit_solutions[0]
    # Reporting (optional)
    if report or return_report:
        best_costs = _summarize_best_costs(solutions)
        if report:
            _print_cost_report(best_costs, chosen=(best[0], best[1], best[2], best[3]))
        if return_report:
            return best[4], best[5], best_costs
    return best[4], best[5]


if __name__ == "__main__":
    pass
