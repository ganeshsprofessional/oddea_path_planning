import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Tuple, Set, Dict
from src.search import SearchResult

class BenchmarkVisualizationModule:
    
    @staticmethod
    def render_and_save_visualizations(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                                       results: List[SearchResult], title_prefix: str, output_dir: str = "results"):
        os.makedirs(output_dir, exist_ok=True)
        colors = {"A*": "blue", "ODDEA*": "red", "Theta*": "green", "Bidirectional A*": "purple"}
        
        # 1. Path Overlay Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid, cmap="binary", origin="upper")
        
        legend_elements = []
        for res in results:
            if res.path:
                ys, xs = zip(*res.path)
                line_color = colors.get(res.algorithm, "orange")
                # Offset lines slightly to prevent them from completely hiding each other
                offset = {"A*": 0.0, "ODDEA*": 0.2, "Theta*": -0.2, "Bidirectional A*": 0.4}.get(res.algorithm, 0)
                ax.plot(np.array(xs) + offset, np.array(ys) + offset, color=line_color, linewidth=2.5, alpha=0.85)
                legend_elements.append(mpatches.Patch(color=line_color, label=f"{res.algorithm} (len: {res.path_length:.2f})"))

        ax.plot(start[1], start[0], marker='o', color='lime', markersize=11, markeredgecolor='black', linestyle='None')
        ax.plot(goal[1], goal[0], marker='*', color='crimson', markersize=14, markeredgecolor='black', linestyle='None')
        legend_elements.extend([
            mpatches.Patch(facecolor='lime', edgecolor='black', label='Start'),
            mpatches.Patch(facecolor='crimson', edgecolor='black', label='Goal')
        ])

        ax.set_title(f"Path Comparison Overlay - {title_prefix}")
        ax.legend(handles=legend_elements, loc="upper right")
        
        path_img_file = os.path.join(output_dir, f"{title_prefix.lower()}_paths.png")
        plt.savefig(path_img_file, dpi=300)
        plt.close()

        # 2. Intuitive Bar Chart Metrics
        algorithms = [r.algorithm for r in results]
        bar_colors = [colors.get(a, "gray") for a in algorithms]
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].bar(algorithms, [r.path_length for r in results], color=bar_colors)
        axs[0, 0].set_title("Path Length")
        
        axs[0, 1].bar(algorithms, [r.execution_time_ms for r in results], color=bar_colors)
        axs[0, 1].set_title("Planning Time (ms)")
        
        axs[1, 0].bar(algorithms, [r.expanded_nodes for r in results], color=bar_colors)
        axs[1, 0].set_title("Expanded Nodes")
        
        axs[1, 1].bar(algorithms, [r.coverage_pct for r in results], color=bar_colors)
        axs[1, 1].set_title("Search Space Coverage (%)")

        plt.suptitle(f"Algorithm Comparison Metrics: {title_prefix}", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{title_prefix.lower()}_metrics_chart.png"), dpi=300)
        plt.close()

    @staticmethod
    def print_metric_table(results: List[SearchResult], title: str):
        print(f"\n{'='*15} {title.upper()} {'='*15}")
        header = f"{'Algorithm':<18} | {'Length':<8} | {'Time (ms)':<9} | {'Nodes':<6} | {'Cov %':<6}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r.algorithm:<18} | {r.path_length:<8.2f} | {r.execution_time_ms:<9.2f} | {r.expanded_nodes:<6} | {r.coverage_pct:<6.2f}")
        print("=" * 55)

    @staticmethod
    def print_statistical_table(stats: Dict[str, Dict[str, float]], title: str):
        print(f"\n{'='*10} {title.upper()} {'='*10}")
        header = f"{'Algorithm':<18} | {'Avg Length':<12} | {'Avg Time (ms)':<14} | {'Avg Nodes':<10} | {'Avg Cov %':<10}"
        print(header)
        print("-" * len(header))
        for algo, metrics in stats.items():
            print(f"{algo:<18} | {metrics['length']:<12.2f} | {metrics['time']:<14.2f} | {metrics['nodes']:<10.2f} | {metrics['cov']:<10.2f}")
        print("=" * 75)