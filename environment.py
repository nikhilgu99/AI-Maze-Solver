import cv2
import numpy as np
from pathlib import Path

class MazeEnv:
    """
    An environment class that processes a .bmp maze image into a navigable grid.
    Walls are assumed to be black pixels, and paths are white pixels.
    """

    def __init__(self, image_path: str | Path):
        self.image_path = Path(image_path)
        self.grid = None
        self.start_node = None
        self.end_node = None
        
        # Initialize the environment upon instantiation
        self._build_environment()

    def _build_environment(self):
        # 1. Load the image in grayscale
        # Using pathlib ensures paths resolve correctly universally
        img = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise FileNotFoundError(f"Could not load image at {self.image_path}.")

        # 2. Convert to a binary grid (0 for walls, 1 for paths)
        _, binary_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        self.grid = binary_img

        # 3. Locate the entrance and exit
        self.start_node, self.end_node = self._find_start_and_end()
        
        print(f"Environment built from {self.image_path.name}")
        print(f"Grid Size: {self.grid.shape[0]}x{self.grid.shape[1]}")
        print(f"Start Node: {self.start_node} | End Node: {self.end_node}")

    def _find_start_and_end(self):
        """
        Scans the top and bottom rows to find the entrance and exit coordinates.
        Returns a tuple of coordinates: ((start_row, start_col), (end_row, end_col))
        """
        rows, cols = self.grid.shape
        
        # Find the single path pixel (value 1) in the top row (row 0)
        start_col = np.where(self.grid[0, :] == 1)[0]
        if len(start_col) == 0:
            raise ValueError("No valid entrance (white pixel) found on the top row.")
        start_node = (0, int(start_col[0]))

        # Find the single path pixel (value 1) in the bottom row (row -1)
        end_col = np.where(self.grid[rows - 1, :] == 1)[0]
        if len(end_col) == 0:
            raise ValueError("No valid exit (white pixel) found on the bottom row.")
        end_node = (rows - 1, int(end_col[0]))

        return start_node, end_node

    def get_valid_neighbors(self, current_node):
        """
        Given a node (row, col), returns a list of valid, walkable adjacent nodes.
        """
        row, col = current_node
        neighbors = []
        
        # Define the 4 cardinal movements: Down, Up, Right, Left
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        rows, cols = self.grid.shape

        for d_row, d_col in directions:
            n_row, n_col = row + d_row, col + d_col
            
            # Check if the neighbor is within the grid bounds
            if 0 <= n_row < rows and 0 <= n_col < cols:
                # Check if the neighbor is a path (1) and not a wall (0)
                if self.grid[n_row, n_col] == 1:
                    neighbors.append((n_row, n_col))
                    
        return neighbors
    
if __name__ == "__main__":
    test_image_path = "test_maze.bmp"
    
    try:
        # Instantiate the environment
        env = MazeEnv(test_image_path)
        
        # Test the neighbor function on a known path pixel (row 2, col 1)
        test_node = (2, 1)
        neighbors = env.get_valid_neighbors(test_node)
        
        print(f"\nTesting get_valid_neighbors for node {test_node}:")
        print(f"Found valid path neighbors at: {neighbors}")
        
    except Exception as e:
        print(f"An error occurred: {e}")