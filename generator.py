import numpy as np
import cv2
import random
from pathlib import Path

def generate_maze(size=100, output_filename="generated_maze.bmp"):
    """
    Generates a random perfect maze using Randomized Prim's Algorithm.
    """
    # Grid dimensions must be odd to allow 1-pixel walls between paths
    if size % 2 == 0: 
        size += 1

    width, height = size, size

    # Initialize the grid with all walls (black)
    grid = np.zeros((height, width), dtype=np.uint8)

    # Pick a random starting point
    start_x = random.randrange(1, width, 2)
    start_y = random.randrange(1, height, 2)
    grid[start_y, start_x] = 255 # Mark as path

    # Initialize the wall list with the neighbors of the starting cell
    walls = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        walls.append((start_x + dx, start_y + dy, start_x, start_y))

    while walls:
        # Pick a random wall from the list
        wall_idx = random.randint(0, len(walls) - 1)
        wx, wy, px, py = walls.pop(wall_idx)

        # Check if the wall is out of grid bounds
        if wx <= 0 or wx >= width - 1 or wy <= 0 or wy >= height - 1:
            continue

        # Determine the node on the opposite side of the wall
        nx, ny = wx + (wx - px), wy + (wy - py)

        # Check bounds for the opposite node
        if nx <= 0 or nx >= width - 1 or ny <= 0 or ny >= height - 1:
            continue

        # If the opposite node is a wall, go through it
        if grid[ny, nx] == 0:
            grid[wy, wx] = 255 # Go through/open the wall
            grid[ny, nx] = 255 # Set the new path cell to white

            # Add the neighboring walls of this new cell to our list
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if grid[ny + dy, nx + dx] == 0:
                    walls.append((nx + dx, ny + dy, nx, ny))

    # Find a valid path in the 2nd row to connect the top entrance
    valid_top_cols = np.where(grid[1, :] == 255)[0]
    entrance_col = random.choice(valid_top_cols)
    grid[0, entrance_col] = 255

    # Find a valid path in the 2nd last row to connect the bottom exit
    valid_bottom_cols = np.where(grid[height - 2, :] == 255)[0]
    exit_col = random.choice(valid_bottom_cols)
    grid[height - 1, exit_col] = 255

    # Save the maze
    output_path = Path(output_filename)
    cv2.imwrite(str(output_path), grid)
    return output_path