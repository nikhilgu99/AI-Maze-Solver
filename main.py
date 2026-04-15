import argparse
from pathlib import Path
from environment import MazeEnv
from search_algorithms import solve_bfs, solve_dfs, solve_a_star
from utils import draw_path_and_save
from q_learning import train_q_learning

def main():
    # Set up the CLI argument parser
    parser = argparse.ArgumentParser(description="AI Maze Solver")

    parser.add_argument(
        "-g", "--generate", 
        type=int, 
        metavar="SIZE",
        help="Generate a new random maze of the specified size"
    )
    
    parser.add_argument(
        "-i", "--image", 
        type=str,
        help="Path to the input .bmp maze image"
    )
    
    parser.add_argument(
        "-a", "--algo", 
        type=str, 
        choices=["bfs", "dfs", "astar", "qlearning"], 
        default="bfs",
        help="Search algorithm to use (currently supports: bfs, dfs, astar, qlearning)"
    )

    parser.add_argument(
        "-he", "--heuristic", 
        type=str, 
        choices=["manhattan", "euclidean"], 
        default="manhattan",
        help="Heuristic to use for A* Search (options: manhattan, euclidean)"
    )

    parser.add_argument(
        "-v", "--visualize", 
        action="store_true",
        help="Watch the search algorithm explore the maze in real-time (disabled for q-learning)"
    )
    
    args = parser.parse_args()

    # Handle maze generation
    if args.generate:
        from generator import generate_maze
        size = args.generate
        output_name = f"random_maze_{size}.bmp"
        print(f"\nGenerating a {size}x{size} maze...")
        generated_path = generate_maze(size=size, output_filename=output_name)
        print(f"Maze successfully generated and saved to: {generated_path}")
        
        # If no image was provided to solve, default to solving the one we just made
        if not args.image:
            args.image = str(generated_path)

    # If generate and image was not provided, show an error
    if not args.image:
        print("\nError: You must provide an --image to solve or use --generate to create one.")
        return
    
    image_path = Path(args.image)
    
    # Verify the file exists
    if not image_path.is_file():
        print(f"Error: The file '{image_path}' does not exist.")
        return

    print(f"\n--- AI Maze Solver ---")
    print(f"Loading image: {image_path.name}")
    print(f"Selected algorithm: {args.algo.upper()}\n")
    
    try:
        # Q-learning uses its own Gym wrapper
        if args.algo == "qlearning":
            if args.visualize:
                print("Note: Real-time visualization is disabled for Q-Learning training.")
            path, q_table = train_q_learning(str(image_path), episodes=20000)
            explored_count = "N/A (Trained via Q-Table)"

        else:
            # Initialize the environment
            env = MazeEnv(image_path)
            
            # Run the selected algorithm
            if args.algo == "bfs":
                path, explored_count = solve_bfs(env, visualize=args.visualize)
            elif args.algo == "dfs":
                path, explored_count = solve_dfs(env, visualize=args.visualize)
            elif args.algo == "astar":
                path, explored_count = solve_a_star(env, heuristic_type=args.heuristic, visualize=args.visualize)
        
        # Output the results
        if path:
            print(f"\nMaze Solved!")
            print(f"    Total steps in optimal path: {len(path)}")
            print(f"    Total nodes explored: {explored_count}")
            output_file = f"{args.algo}_solved_{image_path.name}"
            
            # Draw the path and save
            if draw_path_and_save(image_path, path, output_file):
                print(f"    Visual solution saved to: {output_file}")
        else:
            print("\nNo solution found.")
            
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")

if __name__ == "__main__":
    main()