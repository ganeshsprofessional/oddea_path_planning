#!/usr/bin/env python3
import numpy as np
from src.environment import EnvironmentSetup
from src.search import PathSearchModule
from src.visualization import BenchmarkVisualizationModule

def run_scenario(size: int, label: str, lam: float, beta: float):
    print(f"\n[*] Running {label} (Size {size}x{size})...")
    grid, start, goal = EnvironmentSetup.create_predefined_grid(size)
    
    results = [
        PathSearchModule.run_theta_star(grid, start, goal),
        PathSearchModule.run_standard_astar(grid, start, goal),
        PathSearchModule.run_bidirectional_astar(grid, start, goal),
        PathSearchModule.run_oddea_astar(grid, start, goal, radius=2, lam=lam, beta=beta)
    ]
    
    BenchmarkVisualizationModule.print_metric_table(results, f"{label} Benchmark")
    BenchmarkVisualizationModule.render_and_save_visualizations(grid, start, goal, results, label)

def run_statistical_evaluation(runs: int = 50, size: int = 40):
    print(f"\n[*] Running Statistical Evaluation on {runs} Random {size}x{size} Maps...")
    
    # Store accumulators
    algorithms = ["Theta*", "A*", "Bidirectional A*", "ODDEA*"]
    stats = {alg: {"length": [], "time": [], "nodes": [], "cov": []} for alg in algorithms}

    for i in range(runs):
        grid, start, goal = EnvironmentSetup.create_random_paper_map(size=size, density=0.25, seed=100+i)
        
        res = [
            PathSearchModule.run_theta_star(grid, start, goal),
            PathSearchModule.run_standard_astar(grid, start, goal),
            PathSearchModule.run_bidirectional_astar(grid, start, goal),
            PathSearchModule.run_oddea_astar(grid, start, goal, radius=2, lam=0.6, beta=3.0)
        ]
        
        for r in res:
            if r.path:
                stats[r.algorithm]["length"].append(r.path_length)
                stats[r.algorithm]["time"].append(r.execution_time_ms)
                stats[r.algorithm]["nodes"].append(r.expanded_nodes)
                stats[r.algorithm]["cov"].append(r.coverage_pct)

    # Average the stats
    avg_stats = {}
    for alg in algorithms:
        avg_stats[alg] = {
            "length": np.mean(stats[alg]["length"]),
            "time": np.mean(stats[alg]["time"]),
            "nodes": np.mean(stats[alg]["nodes"]),
            "cov": np.mean(stats[alg]["cov"])
        }

    BenchmarkVisualizationModule.print_statistical_table(avg_stats, f"{runs} Random Maps (40x40)")


if __name__ == "__main__":
    print("[+] Starting Comprehensive ODDEA* Benchmarking Suite...\n")
    
    # 1. Small Map (Table 1 equivalents)
    run_scenario(20, "Grid_20x20", lam=0.5, beta=2.0)
    
    # 2. Medium Map (Table 2 equivalents)
    run_scenario(40, "Grid_40x40", lam=0.6, beta=3.0)
    
    # 3. Large Map (Table 3 equivalents)
    run_scenario(60, "Grid_60x60", lam=0.7, beta=3.0)
    
    # 4. Statistical Tests (Table 4 equivalents)
    run_statistical_evaluation(runs=50, size=40)

    print("\n[✓] Execution complete. Check the 'results/' folder for overlay and bar chart images.")