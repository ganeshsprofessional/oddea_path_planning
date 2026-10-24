import math
import heapq
import time
from typing import List, Tuple, Dict, Set
import numpy as np
from src.heuristic import ODDEAHeuristicEngine

class SearchResult:
    def __init__(self, algorithm: str, path: List[Tuple[int, int]], expanded_nodes: int, 
                 visited_cells: Set[Tuple[int, int]], execution_time_ms: float, 
                 path_length: float, coverage_pct: float, snapshots: List[Set[Tuple[int, int]]] = None):
        self.algorithm = algorithm
        self.path = path
        self.expanded_nodes = expanded_nodes
        self.visited_cells = visited_cells
        self.execution_time_ms = execution_time_ms
        self.path_length = path_length
        self.coverage_pct = coverage_pct
        self.snapshots = snapshots if snapshots else []

class PathSearchModule:
    NEIGHBORS = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))
    ]

    @staticmethod
    def _is_valid(grid: np.ndarray, r: int, c: int) -> bool:
        return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == 0

    @staticmethod
    def _reconstruct_path(parent: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [current]
        while current in parent:
            current = parent[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _compute_path_length(path: List[Tuple[int, int]]) -> float:
        if not path or len(path) < 2: return 0.0
        return sum(math.dist(path[i], path[i+1]) for i in range(len(path) - 1))

    @staticmethod
    def _line_of_sight(grid: np.ndarray, s1: Tuple[int, int], s2: Tuple[int, int]) -> bool:
        """Bresenham's line-of-sight for Theta*."""
        x0, y0 = s1
        x1, y1 = s2
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2; dy *= 2

        for _ in range(n):
            if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]): return False
            if grid[x, y] == 1: return False
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return True

    @classmethod
    def run_standard_astar(cls, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> SearchResult:
        start_time = time.perf_counter()
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        g_score = {start: 0.0}
        parent, visited = {}, set()
        expanded_nodes = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited: continue
            visited.add(current)
            expanded_nodes += 1
            if current == goal: break

            for dr, dc, cost in cls.NEIGHBORS:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)
                if cls._is_valid(grid, nr, nc) and neighbor not in visited:
                    tentative_g = g_score[current] + cost
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        g_score[neighbor] = tentative_g
                        h = max(abs(nr - goal[0]), abs(nc - goal[1]))
                        parent[neighbor] = current
                        heapq.heappush(open_set, (tentative_g + h, neighbor))

        path = cls._reconstruct_path(parent, goal) if goal in visited else []
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        coverage = (len(visited) / grid.size) * 100.0
        return SearchResult("A*", path, expanded_nodes, visited, elapsed_ms, cls._compute_path_length(path), coverage)

    @classmethod
    def run_oddea_astar(cls, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                        radius: int = 3, lam: float = 0.5, beta: float = 1.2, record_snapshots: bool = False) -> SearchResult:
        start_time = time.perf_counter()
        engine = ODDEAHeuristicEngine(grid, radius=radius, lam=lam, beta=beta)
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        g_score = {start: 0.0}
        parent, visited = {}, set()
        expanded_nodes = 0
        snapshots = []

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited: continue
            visited.add(current)
            expanded_nodes += 1

            if record_snapshots and expanded_nodes % 50 == 0:
                snapshots.append(set(visited))

            if current == goal:
                if record_snapshots: snapshots.append(set(visited))
                break

            for dr, dc, cost in cls.NEIGHBORS:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)
                if cls._is_valid(grid, nr, nc) and neighbor not in visited:
                    tentative_g = g_score[current] + cost
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        g_score[neighbor] = tentative_g
                        h_prime = engine.calculate_heuristic(neighbor, goal)
                        parent[neighbor] = current
                        heapq.heappush(open_set, (tentative_g + h_prime, neighbor))

        path = cls._reconstruct_path(parent, goal) if goal in visited else []
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        coverage = (len(visited) / grid.size) * 100.0
        return SearchResult("ODDEA*", path, expanded_nodes, visited, elapsed_ms, cls._compute_path_length(path), coverage, snapshots)

    @classmethod
    def run_theta_star(cls, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> SearchResult:
        start_time = time.perf_counter()
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        g_score = {start: 0.0}
        parent = {start: start}
        visited = set()
        expanded_nodes = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited: continue
            visited.add(current)
            expanded_nodes += 1
            if current == goal: break

            for dr, dc, cost in cls.NEIGHBORS:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)
                if cls._is_valid(grid, nr, nc) and neighbor not in visited:
                    p = parent[current]
                    if cls._line_of_sight(grid, p, neighbor):
                        tentative_g = g_score[p] + math.dist(p, neighbor)
                        if tentative_g < g_score.get(neighbor, float('inf')):
                            g_score[neighbor] = tentative_g
                            parent[neighbor] = p
                            h = max(abs(nr - goal[0]), abs(nc - goal[1]))
                            heapq.heappush(open_set, (tentative_g + h, neighbor))
                    else:
                        tentative_g = g_score[current] + cost
                        if tentative_g < g_score.get(neighbor, float('inf')):
                            g_score[neighbor] = tentative_g
                            parent[neighbor] = current
                            h = max(abs(nr - goal[0]), abs(nc - goal[1]))
                            heapq.heappush(open_set, (tentative_g + h, neighbor))
        
        path = []
        if goal in visited:
            curr = goal
            while curr != start:
                path.append(curr)
                curr = parent[curr]
            path.append(start)
            path.reverse()
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        coverage = (len(visited) / grid.size) * 100.0
        return SearchResult("Theta*", path, expanded_nodes, visited, elapsed_ms, cls._compute_path_length(path), coverage)

    @classmethod
    def run_bidirectional_astar(cls, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> SearchResult:
        start_time = time.perf_counter()
        open_f, open_b = [], []
        heapq.heappush(open_f, (0.0, start))
        heapq.heappush(open_b, (0.0, goal))
        g_f, g_b = {start: 0.0}, {goal: 0.0}
        parent_f, parent_b = {}, {}
        visited_f, visited_b = set(), set()
        expanded_nodes = 0
        meeting_node = None

        while open_f and open_b:
            # Forward
            _, curr_f = heapq.heappop(open_f)
            if curr_f not in visited_f:
                visited_f.add(curr_f)
                expanded_nodes += 1
                if curr_f in visited_b:
                    meeting_node = curr_f
                    break
                for dr, dc, cost in cls.NEIGHBORS:
                    nr, nc = curr_f[0] + dr, curr_f[1] + dc
                    n_node = (nr, nc)
                    if cls._is_valid(grid, nr, nc) and n_node not in visited_f:
                        tentative_g = g_f[curr_f] + cost
                        if tentative_g < g_f.get(n_node, float('inf')):
                            g_f[n_node] = tentative_g
                            parent_f[n_node] = curr_f
                            heapq.heappush(open_f, (tentative_g + max(abs(nr - goal[0]), abs(nc - goal[1])), n_node))

            # Backward
            _, curr_b = heapq.heappop(open_b)
            if curr_b not in visited_b:
                visited_b.add(curr_b)
                expanded_nodes += 1
                if curr_b in visited_f:
                    meeting_node = curr_b
                    break
                for dr, dc, cost in cls.NEIGHBORS:
                    nr, nc = curr_b[0] + dr, curr_b[1] + dc
                    n_node = (nr, nc)
                    if cls._is_valid(grid, nr, nc) and n_node not in visited_b:
                        tentative_g = g_b[curr_b] + cost
                        if tentative_g < g_b.get(n_node, float('inf')):
                            g_b[n_node] = tentative_g
                            parent_b[n_node] = curr_b
                            heapq.heappush(open_b, (tentative_g + max(abs(nr - start[0]), abs(nc - start[1])), n_node))

        all_visited = visited_f.union(visited_b)
        path = []
        if meeting_node:
            p_f = cls._reconstruct_path(parent_f, meeting_node)
            curr = meeting_node
            p_b = []
            while curr in parent_b:
                curr = parent_b[curr]
                p_b.append(curr)
            path = p_f + p_b

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        coverage = (len(all_visited) / grid.size) * 100.0
        return SearchResult("Bidirectional A*", path, expanded_nodes, all_visited, elapsed_ms, cls._compute_path_length(path), coverage)