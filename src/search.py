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
        if not path or len(path) < 2:
            return 0.0
        return sum(math.dist(path[i], path[i+1]) for i in range(len(path) - 1))

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

            # Capture intermediate state for final output requirement
            if record_snapshots and expanded_nodes % 50 == 0:
                snapshots.append(set(visited))

            if current == goal:
                if record_snapshots:
                    snapshots.append(set(visited))
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