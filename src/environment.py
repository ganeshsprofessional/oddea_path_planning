import numpy as np
import random
from typing import Tuple, Optional

class EnvironmentSetup:
    """Handles grid map generation, obstacle placement, and start/goal validation."""
    
    @staticmethod
    def create_grid(size: int, obstacle_density: float = 0.2, seed: Optional[int] = 42) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        grid = (np.random.rand(size, size) < obstacle_density).astype(int)
        return grid

    @staticmethod
    def create_predefined_grid(size: int) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        grid = np.zeros((size, size), dtype=int)
        
        if size == 20:
            grid[3:17, 7] = 1
            grid[3:17, 13] = 1
            grid[6, 7] = 0   
            grid[14, 13] = 0 
        elif size == 40:
            grid[5:35, 12] = 1
            grid[5:35, 26] = 1
            grid[10, 12] = 0
            grid[30, 26] = 0
            grid[20, 15:24] = 1
        elif size == 60:
            grid[5:55, 18] = 1
            grid[5:55, 38] = 1
            grid[15, 18] = 0
            grid[45, 38] = 0
            grid[30, 18:38] = 1
            grid[30, 28] = 0
            grid[10:18, 45:52] = 1
        
        start = (1, 1)
        goal = (size - 2, size - 2)
        grid[start] = 0
        grid[goal] = 0
        return grid, start, goal

    @staticmethod
    def validate_or_sample_start_goal(grid: np.ndarray, seed: Optional[int] = None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if seed is not None:
            random.seed(seed)
        size = grid.shape[0]
        free_cells = [(r, c) for r in range(size) for c in range(size) if grid[r, c] == 0]
        start = free_cells[0]
        goal = free_cells[-1]
        return start, goal