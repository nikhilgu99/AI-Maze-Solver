# AI-Maze-Solver
CS5100 AI Maze Solver Project

### Available Options

##### To generate a maze, use the -g or --generate flag:
```
python main.py -g 101
```

##### To solve a maze, use the -a or --algorithm flag with an option (bfs/dfs/astar/qlearning):
```
python main.py -g 101 -a dfs
```
NOTE: This generates a maze and directly solves it at once  
NOTE: To choose a heuristic for astar, add the -he or --heuristic flag (manhattan/euclidean)


##### To use your own maze, pass it with the -i or --image flag:
```
python main.py -i maze_101.bmp -a astar
```



##### To visualize any of the traditional search algorithms, use the -v or --visualize flag:
```
python main.py -i maze_101.bmp -a dfs -v
```
NOTE: This will significantly slow down execution time!



##### To benchmark the algorithms, run benchmark.py, this will generate mazes and plot results automatically:
```
python benchmark.py
```



### Project Structure:
- main.py (The entry point)
- environment.py (The image-to-grid parser)
- search_algorithms.py (BFS, DFS, A*)
- generator.py (Prim's Algorithm maze generator)
- rl_environment.py (The Gymnasium wrapper)
- q_learning.py (The RL agent and training loop)
- benchmark.py (The automated tester)
- utils.py (Path drawing and image saving)



### Other Notes
- The benchmark folder contains a sample output, complete with the generated mazes and benchmark results. I have added the solved versions here to show the solutions.
- In order to run this project, a few external libraries are required, they can be installed with: `pip install numpy opencv-python gymnasium pandas matplotlib`
