import math
import numpy as np
from typing import Tuple, Dict

class ODDEAHeuristicEngine:
    """Computes density-aware dynamic exponential heuristic."""
    def __init__(self, grid: np.ndarray, radius: int = 3, lam: float = 0.5, beta: float = 1.2):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.radius = radius
        self.lam = lam
        self.beta = beta
        self.density_cache: Dict[Tuple[int, int], float] = {}

    def get_chebyshev_distance(self, n: Tuple[int, int], goal: Tuple[int, int]) -> float:
        dx = abs(n[0] - goal[0])
        dy = abs(n[1] - goal[1])
        return max(dx, dy)

    def compute_local_density(self, n: Tuple[int, int]) -> float:
        if n in self.density_cache:
            return self.density_cache[n]

        r, c = n
        r_min, r_max = max(0, r - self.radius), min(self.rows - 1, r + self.radius)
        c_min, c_max = max(0, c - self.radius), min(self.cols - 1, c + self.radius)

        subgrid = self.grid[r_min:r_max + 1, c_min:c_max + 1]
        total_cells = subgrid.size
        obstacle_count = np.count_nonzero(subgrid == 1)

        density = obstacle_count / total_cells if total_cells > 0 else 0.0
        self.density_cache[n] = density
        return density

    def calculate_heuristic(self, n: Tuple[int, int], goal: Tuple[int, int]) -> float:
        base_h = self.get_chebyshev_distance(n, goal)
        d_n = self.compute_local_density(n)
        weight = 1.0 + self.lam * math.exp(self.beta * d_n)
        return weight * base_h