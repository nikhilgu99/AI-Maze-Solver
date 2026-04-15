import time
#import csv
import pandas as pd
import matplotlib.pyplot as plt
#from pathlib import Path

from generator import generate_maze
from environment import MazeEnv
from search_algorithms import solve_bfs, solve_dfs, solve_a_star
from q_learning import train_q_learning

def run_benchmarks():
    # Phase 1: All Algorithms (Capped at 71 to keep Q-Learning feasible)
    comparison_sizes = [11, 31, 51, 71]
    
    # Phase 2: Traditional Search Stress Test (Larger Mazes)
    stress_sizes = [101, 501, 1001, 5001]
    
    results = []

    print("--- Phase 1: Head-to-Head Comparison ---")
    for size in comparison_sizes:
        print(f"\nGenerating {size}x{size} maze...")
        maze_file = generate_maze(size=size, output_filename=f"bench_maze_{size}.bmp")
        
        # BFS
        env = MazeEnv(maze_file)
        start = time.time()
        path_bfs, nodes_bfs = solve_bfs(env)
        time_bfs = time.time() - start
        
        # DFS
        env = MazeEnv(maze_file)
        start = time.time()
        path_dfs, nodes_dfs = solve_dfs(env)
        time_dfs = time.time() - start
        
        # A*
        env = MazeEnv(maze_file)
        start = time.time()
        path_astar, nodes_astar = solve_a_star(env, heuristic_type="manhattan")
        time_astar = time.time() - start
        
        # Q-Learning
        episodes = size * 300
        print(f"    Training Q-Learning for {episodes} episodes...")
        start = time.time()
        path_ql, _ = train_q_learning(str(maze_file), episodes=episodes)
        time_ql = time.time() - start
        
        ql_optimality = "Optimal" if path_ql and len(path_ql) <= len(path_astar) else "Suboptimal/Failed"

        results.append({
            "Maze Size": f"{size}x{size}",
            "BFS Time": time_bfs, "DFS Time": time_dfs, "A* Time": time_astar,
            "BFS Nodes": nodes_bfs, "DFS Nodes": nodes_dfs, "A* Nodes": nodes_astar,
            "QL Time": time_ql, "QL Status": ql_optimality
        })
        print(f"    {size}x{size} complete. A* Time: {time_astar:.4f}s | QL Time: {time_ql:.4f}s")


    print("\n--- Phase 2: Traditional Search Stress Test ---")
    for size in stress_sizes:
        print(f"\n[*] Generating larger {size}x{size} maze...")
        maze_file = generate_maze(size=size, output_filename=f"bench_maze_{size}.bmp")
        
        env = MazeEnv(maze_file)
        start = time.time()
        _, nodes_bfs = solve_bfs(env)
        time_bfs = time.time() - start
        
        env = MazeEnv(maze_file)
        start = time.time()
        _, nodes_dfs = solve_dfs(env)
        time_dfs = time.time() - start
        
        env = MazeEnv(maze_file)
        start = time.time()
        _, nodes_astar = solve_a_star(env, heuristic_type="manhattan")
        time_astar = time.time() - start
        
        results.append({
            "Maze Size": f"{size}x{size}",
            "BFS Time": time_bfs, "DFS Time": time_dfs, "A* Time": time_astar,
            "BFS Nodes": nodes_bfs, "DFS Nodes": nodes_dfs, "A* Nodes": nodes_astar,
            "QL Time": None, "QL Status": "Skipped (Too Large)"
        })
        print(f"    {size}x{size} complete. BFS: {time_bfs:.2f}s | A*: {time_astar:.2f}s")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\nData saved to benchmark_results.csv")

    # Generate Graph
    plt.figure(figsize=(10, 6))
    
    all_sizes = comparison_sizes + stress_sizes
    all_labels = [f"{s}x{s}" for s in all_sizes]
    
    plt.plot(all_labels, df["BFS Time"], marker='o', label="BFS Time", color="blue")
    plt.plot(all_labels, df["DFS Time"], marker='s', label="DFS Time", color="red")
    plt.plot(all_labels, df["A* Time"], marker='^', label="A* Time", color="green")
    
    plt.title("Execution Time Scaling With Maze Sizes")
    plt.xlabel("Maze Dimensions")
    plt.ylabel("Execution Time (Seconds)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("benchmark_graph_time.png")
    print("Comparison graph saved to benchmark_graph_time.png")
    plt.show()

if __name__ == "__main__":
    run_benchmarks()