# FrozenLake Reinforcement Learning Agent

## Project Overview
This project implements a Q-Learning reinforcement learning agent using the FrozenLake-v1 environment from Gymnasium.

The agent learns to move across a frozen grid and reach the goal while avoiding holes.

## Algorithm
**Q-Learning** is a model-free reinforcement learning algorithm that learns action values through trial and error.

The update rule is:

Q(s,a) = Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]

## Environment
- Environment: FrozenLake-v1
- States: 16
- Actions: 4 (Left, Down, Right, Up)
- Learning rate: 0.8
- Discount factor: 0.95
- Training episodes: 15,000
- Maximum steps per episode: 100

## Project Files
- `main.py` — Main training program
- `agent.py` — Q-Learning agent implementation
- `utils.py` — Agent evaluation utility
- `requirements.txt` — Required Python packages
- `q_table.npy` — Saved Q-table
- `training_progress.png` — Training reward plot
- `q_table.png` — Q-table visualization
- `trained_agent.gif` — Trained agent demonstration
- `.ipynb` — Executed Jupyter notebook

## How to Run
Install the required packages:

`pip install -r requirements.txt`

Run the program:

`python main.py`

## Results
The trained agent successfully reached the goal during the demonstration. The training graph and saved Q-table are included in this repository.

## Conclusion
This project demonstrates how Q-Learning can train an agent to learn a policy in a reinforcement learning environment.
