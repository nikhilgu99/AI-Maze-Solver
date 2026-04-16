import numpy as np
import random
from rl_environment import GymMazeEnv

def train_q_learning(image_path, episodes=2000):
    """
    Trains a Q-learning agent on the given maze image.
    Returns the solved path coordinates and the final Q-table.
    """
    env = GymMazeEnv(image_path)
    
    # Initialize the Q-Table with zeros
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    alpha = 0.1          # Learning Rate
    gamma = 0.99         # Discount Factor
    epsilon = 1.0        # Exploration Rate
    epsilon_decay = 0.999 # How fast epsilon drops per episode
    min_epsilon = 0.01   # Minimum exploration rate
    
    print(f"Training Q-Learning Agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        terminated = False
        
        while not terminated:
            # Epsilon-Greedy Action Selection
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore: Pick a random action
            else:
                action = np.argmax(q_table[state, :]) # Exploit: Pick the best known action
                
            # Take the action
            next_state, reward, terminated, _, _ = env.step(action)
            
            # Update the Q-value using Bellman Equation
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state, :])
            
            # Q(s,a) = Q(s,a) + alpha * [Reward + gamma * max(Q(s',a')) - Q(s,a)]
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state, action] = new_value
            
            state = next_state
            
        # Decay Epsilon at the end of each episode
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if (episode + 1) % 500 == 0:
            print(f"  Episode {episode + 1}/{episodes} completed.")
            
    print("Training Complete.")
    
    # --- Extract the Optimal Path ---
    path = []
    state, _ = env.reset()
    env.current_pos = env.maze.start_node
    
    # Run the trained agent with no random actions
    while env.current_pos != env.maze.end_node:
        path.append(env.current_pos)
        action = np.argmax(q_table[state, :]) # Always exploit
        state, _, terminated, _, _ = env.step(action)
        
        # Failsafe in case it didn't learn a complete path
        if len(path) > (env.rows * env.cols):
            print("[!] Agent stuck in a loop. Training insufficient.")
            return None, q_table
            
    path.append(env.maze.end_node)
    return path, q_table