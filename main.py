#!/usr/bin/env python3
from src.environment import EnvironmentSetup
from src.search import PathSearchModule
from src.visualization import BenchmarkVisualizationModule

def run_benchmark_and_snapshots():
    print("[+] Initializing ODDEA* Project Environment...")
    
    # Generate Medium Grid (40x40)
    grid, start, goal = EnvironmentSetup.create_predefined_grid(40)
    
    print("[*] Running Standard A*...")
    res_a = PathSearchModule.run_standard_astar(grid, start, goal)
    
    print("[*] Running ODDEA* (with intermediate snapshot recording enabled)...")
    res_oddea = PathSearchModule.run_oddea_astar(grid, start, goal, record_snapshots=True)
    
    results = [res_a, res_oddea]
    
    # Print Metrics
    BenchmarkVisualizationModule.print_metric_table(results, "Grid 40x40 Benchmark")
    
    # Save Final Overlays
    BenchmarkVisualizationModule.render_and_save_visualizations(grid, start, goal, results, "Grid_40x40")
    
    # Save Intermediate Snapshots for the Implementation Document
    print("[*] Rendering intermediate search snapshots...")
    BenchmarkVisualizationModule.render_intermediate_snapshots(grid, start, goal, res_oddea.snapshots)

if __name__ == "__main__":
    run_benchmark_and_snapshots()
    print("\n[✓] Execution complete. Check the 'results/' folder for output and snapshots.")