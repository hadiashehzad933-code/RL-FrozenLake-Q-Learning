import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from agent import QLearningAgent
from utils import evaluate_agent


def main():
    # Create the environment
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False
    )

    # Initialize the agent
    agent = QLearningAgent(
        env.observation_space.n,
        env.action_space.n,
        learning_rate=0.8,
        discount_factor=0.95
    )

    episodes = 15000
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9995
    rewards = []

    # Train the agent
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0

        for step in range(100):
            action = agent.choose_action(
                state, epsilon, env.action_space
            )

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            agent.learn(
                state, action, reward,
                next_state, terminated
            )

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)
        epsilon = max(
            epsilon_min, epsilon * epsilon_decay
        )

    # Save trained model
    np.save("q_table.npy", agent.q_table)

    # Evaluate the agent
    success_rate = evaluate_agent(
        env, agent.q_table, episodes=100
    )

    print("Training completed!")
    print(f"Test success rate: {success_rate:.2f}%")

    # Plot training progress
    block_size = 100
    values = np.array(rewards)
    averages = values.reshape(-1, block_size).mean(axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(
        np.arange(1, len(averages) + 1) * block_size,
        averages
    )
    plt.xlabel("Training Episodes")
    plt.ylabel("Average Reward")
    plt.title("Q-Learning Training Progress")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=150)
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
