import math
import heapq
import cv2
import numpy as np
from collections import deque

def setup_visualization(env):
    """Creates a color copy of the binary grid for OpenCV rendering."""
    # Convert the 0/1 grid to a 0/255 grayscale, then to BGR color
    display_img = cv2.cvtColor(env.grid * 255, cv2.COLOR_GRAY2BGR)
    
    # Calculate a scale factor so tiny mazes scale up to roughly 500x500 pixels
    scale = max(1, 500 // max(env.grid.shape))
    return display_img, scale

def render_frame(display_img, scale, current, nodes_explored):
    """Updates the image and renders it to the screen."""
    # Color the current visited node gray
    display_img[current[0], current[1]] = (0, 0, 200)
    
    # Only update the GUI every 5 nodes to keep it faster/smoother
    if nodes_explored % 5 == 0:
        render_img = cv2.resize(display_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Algorithm Animation", render_img)
        cv2.waitKey(1)

def solve_bfs(env, visualize=False):
    """
    Solves the maze using Breadth-First Search (BFS).
    Takes a MazeEnv instance as input.
    Returns a tuple: (path_coordinates_list, nodes_explored_count)
    """
    start = env.start_node
    goal = env.end_node
    
    # Queue for BFS
    queue = deque([start])
    
    # Dictionary to track visited nodes and reconstruct the path.
    # Key = current node, Value = the node we came from
    came_from = {start: None}
    
    nodes_explored = 0
    
    if visualize:
        display_img, scale = setup_visualization(env)
    
    while queue:
        current = queue.popleft()
        nodes_explored += 1
        
        if visualize and current != start and current != goal:
            render_frame(display_img, scale, current, nodes_explored)
        
        # If we reach the exit, stop searching
        if current == goal:
            break
            
        # Ask the environment for valid next moves
        for neighbor in env.get_valid_neighbors(current):
            # If we haven't visited this neighbor yet
            if neighbor not in came_from:
                queue.append(neighbor)
                came_from[neighbor] = current  # Record where we came from
                
    if visualize:
        cv2.destroyAllWindows()
                
    # Reconstruct the final path by backtracking from the goal
    path = []
    if goal in came_from:
        current_step = goal
        while current_step is not None:
            path.append(current_step)
            current_step = came_from[current_step]
            
        # Reverse the path so it goes from start -> end
        path.reverse()
        
    return path, nodes_explored

def solve_dfs(env, visualize=False):
    """
    Solves the maze using Depth-First Search (DFS).
    Takes a MazeEnv instance as input.
    Returns a tuple: (path_coordinates_list, nodes_explored_count)
    """
    start = env.start_node
    goal = env.end_node
    
    # Stack for DFS
    stack = [start]
    
    # Dictionary to track visited nodes and reconstruct the path
    came_from = {start: None}
    
    nodes_explored = 0
    
    if visualize:
        display_img, scale = setup_visualization(env)
    
    while stack:
        # Removes and returns the LAST element added
        current = stack.pop()
        nodes_explored += 1
        
        if visualize and current != start and current != goal:
            render_frame(display_img, scale, current, nodes_explored)
        
        # Reached the exit, stop searching
        if current == goal:
            break
            
        # Ask the environment for valid next moves
        for neighbor in env.get_valid_neighbors(current):
            # If we haven't visited this neighbor yet
            if neighbor not in came_from:
                stack.append(neighbor)
                came_from[neighbor] = current  # Record where we came from
                
    if visualize:
        cv2.destroyAllWindows()
                
    # Reconstruct the final path by backtracking from the goal
    path = []
    if goal in came_from:
        current_step = goal
        while current_step is not None:
            path.append(current_step)
            current_step = came_from[current_step]
            
        # Reverse the path so it goes from start -> end
        path.reverse()
        
    return path, nodes_explored

def manhattan_distance(node1, node2):
    """Calculates the Manhattan distance between two nodes on a grid."""
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

def euclidean_distance(node1, node2):
    """Calculates the Euclidean (straight-line) distance between two nodes."""
    return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

def solve_a_star(env, heuristic_type="manhattan", visualize=False):
    """
    Solves the maze using A* Search.
    Takes a MazeEnv instance and a heuristic_type ("manhattan" or "euclidean").
    Returns a tuple: (path_coordinates_list, nodes_explored_count)
    """
    start = env.start_node
    goal = env.end_node
    
    # Priority queue for A* stores tuples of: (f_score, tie_breaker_count, node)
    open_set = []
    counter = 0 # Tie-breaker ensures we don't compare tuples if f_scores are equal
    heapq.heappush(open_set, (0, counter, start))
    
    came_from = {start: None}
    # g_score tracks the exact cost from the start node to the current node
    g_score = {start: 0}
    nodes_explored = 0
    
    if visualize:
        display_img, scale = setup_visualization(env)
    
    while open_set:
        # Pop the node with the lowest f_score (g_score + heuristic)
        current_f, _, current = heapq.heappop(open_set)
        nodes_explored += 1
        
        if visualize and current != start and current != goal:
            render_frame(display_img, scale, current, nodes_explored)
        
        # Reached the exit, stop searching
        if current == goal:
            break
            
        for neighbor in env.get_valid_neighbors(current):
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                
                # Calculate the heuristic based on the user's choice
                if heuristic_type == "euclidean":
                    h_cost = euclidean_distance(neighbor, goal)
                else:
                    h_cost = manhattan_distance(neighbor, goal)
                
                f_score = tentative_g_score + h_cost
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))
                
    if visualize:
        cv2.destroyAllWindows()
                
    path = []
    if goal in came_from:
        current_step = goal
        while current_step is not None:
            path.append(current_step)
            current_step = came_from[current_step]
        path.reverse()
        
    return path, nodes_explored

if __name__ == "__main__":
    from environment import MazeEnv
    
    try:
        # Load the test maze
        test_env = MazeEnv("test_maze.bmp")
        
        print("\nRunning BFS...")
        # Set visualize to True here for testing
        path, explored_count = solve_bfs(test_env, visualize=False)
        
        if path:
            print(f"Success! Path found in {len(path)} steps.")
            print(f"Total nodes explored to find path: {explored_count}")
            print(f"Path coordinates: {path}")
        else:
            print("No path found.")
            
    except Exception as e:
        print(f"An error occurred: {e}")