import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        learning_rate=0.8,
        discount_factor=0.95
    ):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount_factor

    def choose_action(self, state, epsilon, action_space):
        if np.random.random() < epsilon:
            return action_space.sample()

        return int(np.argmax(self.q_table[state]))

    def learn(
        self, state, action, reward,
        next_state, terminated
    ):
        if terminated:
            future_reward = 0
        else:
            future_reward = (
                self.gamma * np.max(self.q_table[next_state])
            )

        target = reward + future_reward

        self.q_table[state, action] += self.lr * (
            target - self.q_table[state, action]
        )
