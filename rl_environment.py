import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment import MazeEnv

class GymMazeEnv(gym.Env):
    """
    A Gymnasium wrapper for the existing MazeEnv.
    Allows a Q-Learning agent to interact with the parsed .bmp maze.
    """
    def __init__(self, image_path):
        super(GymMazeEnv, self).__init__()
        
        # Load existing image processing environment
        self.maze = MazeEnv(image_path)
        self.grid = self.maze.grid
        self.rows, self.cols = self.grid.shape
        
        # Action Space: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: A 1D integer representing the current (row, col)
        self.observation_space = spaces.Discrete(self.rows * self.cols)
        
        # Directions corresponding to actions
        self.action_to_direction = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        self.current_pos = self.maze.start_node

    def _get_state(self, pos):
        """Flattens a 2D (row, col) coordinate into a 1D state integer."""
        row, col = pos
        return row * self.cols + col

    def reset(self, seed=None, options=None):
        """Resets the agent to the starting position."""
        super().reset(seed=seed)
        self.current_pos = self.maze.start_node
        return self._get_state(self.current_pos), {}

    def step(self, action):
        """Executes one movement and returns the new state and reward."""
        row, col = self.current_pos
        d_row, d_col = self.action_to_direction[action]
        new_row, new_col = row + d_row, col + d_col
        
        # Default flags
        terminated = False
        truncated = False
        reward = 0
        
        # Check if the move is out of bounds or hits a wall (0)
        if (new_row < 0 or new_row >= self.rows or 
            new_col < 0 or new_col >= self.cols or 
            self.grid[new_row, new_col] == 0):
            
            # The agent hit a wall! It stays in the same place and gets penalized.
            reward = -5  # Wall penalty
            
        else:
            # The move is valid
            self.current_pos = (new_row, new_col)
            
            # Check if the agent reached the goal
            if self.current_pos == self.maze.end_node:
                reward = 100  # Large reward for solving maze
                terminated = True
            else:
                # Normal step
                reward = -1  # Small penalty for every step to encourage finding the FASTEST route
                
        state = self._get_state(self.current_pos)
        return state, reward, terminated, truncated, {}