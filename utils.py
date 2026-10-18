import numpy as np


def evaluate_agent(env, q_table, episodes=100):
    successes = 0

    for _ in range(episodes):
        state, info = env.reset()

        for _ in range(100):
            action = int(np.argmax(q_table[state]))

            state, reward, terminated, truncated, info = (
                env.step(action)
            )

            if terminated or truncated:
                if reward == 1:
                    successes += 1
                break

    return (successes / episodes) * 100
