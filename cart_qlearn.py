import gym
import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, env_name, env_space, action_space, env_obs_max, env_obs_min=None):
        self.env_name = env_name
        self.env_space = env_space
        self.action_space = action_space
        self.env_obs_max = env_obs_max
        self.env_obs_min = env_obs_max * -1 if env_obs_min is None else env_obs_min

        self.n_episode = 25000
        self.show_every = 5000

        self.lr = 0.1
        self.discount_rate = 0.95

        self.epsilon = 1
        self.epsilon_decay_rate = 0.995
        self.epsilon_min = 0.05

        self.n_splits = 16
        self.splits_space = np.linspace(self.env_obs_min, self.env_obs_max, self.n_splits).T
        self.q_table = np.random.uniform(0, self.action_space, [self.n_splits] * self.env_space + [self.action_space])

    def get_discrete_state(self, state):
        idx = [
            np.digitize(state[i], self.splits_space[i]) - 1 for i in range(len(state))
        ]

        return tuple(idx)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(
                self.q_table[self.get_discrete_state(state)]
            )

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon * self.epsilon_decay_rate,
            self.epsilon_min
        )

    def update_q_table(self, current_state, future_state, action, reward):
        current_discrete_state = self.get_discrete_state(current_state)
        future_discrete_state = self.get_discrete_state(future_state)

        current_q = self.q_table[current_discrete_state + (action, )]
        max_future_q = np.max(self.q_table[future_discrete_state])
        new_q = (1.0 - self.lr) * current_q + self.lr * (reward + self.discount_rate * max_future_q)

        self.q_table[current_discrete_state + (action, )] = new_q

    def train(self):
        env = gym.make(self.env_name)
        history = np.zeros(self.n_episode)

        for episode in range(self.n_episode):
            episode_reward = 0
            current_state = env.reset()

            for i in range(200):
                # if episode % self.show_every == 0:
                #     env.render()

                action = self.get_action(current_state)
                future_state, reward, done, _ = env.step(action)
                episode_reward += reward

                # modify policy, allow model to converge easier
                if not done:
                    if abs(future_state[0]) < 4:
                        reward += 2
                    if abs(future_state[2]) < 0.3:
                        reward += 2
                else:
                    if episode_reward < 200:
                        reward = -200
                self.update_q_table(current_state, future_state, action, reward)

                current_state = future_state

                if done:
                    break

            self.decay_epsilon()
            history[episode] = episode_reward

            if episode >= 100:
                print(episode, episode_reward, np.average(history[episode-100:episode]))
            else:
                print(episode, episode_reward)

        env.close()

        plt.plot(np.arange(self.n_episode), history, linewidth=0.1)
        plt.plot(np.arange(self.n_episode-100+1)+100//2, np.convolve(history, np.ones(100)/100.0, "valid"))
        plt.show()

    def render_result(self):
        env = gym.make(self.env_name)

        for _ in range(10):
            done = False
            current_state = env.reset()
            while not done:
                env.render()
                action = self.get_action(current_state)
                current_state, reward, done, _ = env.step(action)

        env.close()


if __name__ == "__main__":
    agent = Agent("CartPole-v0", 4, 2, np.array([4.8, 1.0, 0.418, 2.0]))
    agent.train()
    # agent.render_result()
