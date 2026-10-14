import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Tuple, Set
from src.search import SearchResult

class BenchmarkVisualizationModule:
    
    @staticmethod
    def render_and_save_visualizations(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                                       results: List[SearchResult], title_prefix: str, output_dir: str = "results"):
        os.makedirs(output_dir, exist_ok=True)
        colors = {"A*": "blue", "ODDEA*": "red"}
        
        # 1. Path Overlay Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid, cmap="binary", origin="upper")
        
        # Track custom legend elements to bypass Python 3.14 / Matplotlib recursion bugs
        legend_elements = []
        
        for res in results:
            if res.path:
                ys, xs = zip(*res.path)
                line_color = colors.get(res.algorithm, "orange")
                ax.plot(xs, ys, color=line_color, linewidth=2.5, alpha=0.85)
                # Add a proxy patch for the legend
                legend_elements.append(mpatches.Patch(color=line_color, label=f"{res.algorithm} (len: {res.path_length:.2f})"))

        # Plot Start and Goal without adding them automatically to the legend
        ax.plot(start[1], start[0], marker='o', color='lime', markersize=11, markeredgecolor='black', linestyle='None')
        ax.plot(goal[1], goal[0], marker='*', color='crimson', markersize=14, markeredgecolor='black', linestyle='None')
        
        # Add proxy patches for Start and Goal
        legend_elements.append(mpatches.Patch(facecolor='lime', edgecolor='black', label='Start'))
        legend_elements.append(mpatches.Patch(facecolor='crimson', edgecolor='black', label='Goal'))

        ax.set_title(f"Path Comparison Overlay - {title_prefix}")
        
        # Feed the custom, marker-free handles directly to the legend
        ax.legend(handles=legend_elements, loc="upper right")
        
        path_img_file = os.path.join(output_dir, f"{title_prefix.lower()}_paths.png")
        plt.savefig(path_img_file, dpi=300)
        plt.close()

    @staticmethod
    def render_intermediate_snapshots(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], 
                                      snapshots: List[Set[Tuple[int, int]]], output_dir: str = "results/snapshots"):
        os.makedirs(output_dir, exist_ok=True)
        
        for i, visited_nodes in enumerate(snapshots):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(grid, cmap="binary", origin="upper")
            
            if visited_nodes:
                vy, vx = zip(*visited_nodes)
                ax.scatter(vx, vy, c='cyan', s=10, alpha=0.6)
            
            ax.plot(start[1], start[0], marker='o', color='lime', markersize=10, markeredgecolor='black', linestyle='None')
            ax.plot(goal[1], goal[0], marker='*', color='crimson', markersize=12, markeredgecolor='black', linestyle='None')
            
            ax.set_title(f"ODDEA* Search Space Expansion - Step {i+1}")
            snap_file = os.path.join(output_dir, f"snapshot_step_{i+1:02d}.png")
            plt.savefig(snap_file, dpi=150)
            plt.close()
            
        print(f"[+] Saved {len(snapshots)} intermediate snapshots to '{output_dir}'")

    @staticmethod
    def print_metric_table(results: List[SearchResult], title: str):
        print(f"\n{'='*15} {title.upper()} {'='*15}")
        header = f"{'Algorithm':<12} | {'Length':<8} | {'Time (ms)':<9} | {'Nodes':<6} | {'Cov %':<6}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r.algorithm:<12} | {r.path_length:<8.2f} | {r.execution_time_ms:<9.2f} | {r.expanded_nodes:<6} | {r.coverage_pct:<6.2f}")
        print("=" * 45)