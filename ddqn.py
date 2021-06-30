# Reference
# https://github.com/sanketsans/openAIenv/blob/master/DQN/LunarLander/LunarLander_DQN.ipynb
# https://github.com/plopd/deep-reinforcement-learning/blob/master/dqn/dqn_agent.py

import os
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class ReplayMemory:
    def __init__(self, max_size, obs_dim):
        self.max_size = max_size
        self.size = 0
        self.pointer = 0
        self.current_state_memory = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.future_state_memory = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.reward_memory = np.zeros([max_size], dtype=np.float32)
        self.done_memory = np.zeros([max_size], dtype=np.int)
        self.action_memory = np.zeros([max_size], dtype=np.int)

    def add_memory(self, current, rwd, future, done, act):
        self.current_state_memory[self.pointer] = current
        self.future_state_memory[self.pointer] = future
        self.reward_memory[self.pointer] = rwd
        self.done_memory[self.pointer] = int(done)
        self.action_memory[self.pointer] = int(act)

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_sample(self, mini_batch_size):
        idx = np.random.choice(self.size, mini_batch_size, replace=False)

        return (
            self.current_state_memory[idx],
            self.future_state_memory[idx],
            self.reward_memory[idx],
            self.action_memory[idx],
            self.done_memory[idx]
        )

    def __len__(self):
        return self.size


class Agent:
    def __init__(self, env_space, action_space):
        self.env_space = env_space
        self.action_space = action_space

        self.t_step = 0
        self.update_every = 4

        self.mini_batch_size = 64
        self.max_memory_len = 50000
        self.memory = ReplayMemory(self.max_memory_len, env_space)

        self.epsilon = 1
        self.epsilon_decay_rate = 0.995
        self.epsilon_min = 0.05

        self.lr = 0.0005
        self.discount_rate = 0.995
        self.tau = 0.001

        self.model_train = self.build_model()
        self.model_target = self.build_model()
        self.model_target.set_weights(
            self.model_train.get_weights()
        )

    def build_model(self):
        model = Sequential([
            Dense(128, activation="relu", input_dim=self.env_space, kernel_initializer="he_uniform"),
            Dense(128, activation="relu", kernel_initializer="he_uniform"),
            Dense(self.action_space, activation="linear", kernel_initializer="he_uniform")
        ])

        model.compile(
            optimizer=Adam(lr=self.lr),
            loss="mse"
        )
        return model

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon * self.epsilon_decay_rate,
            self.epsilon_min
        )

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(
                self.model_train.predict(np.array([state]))[0]
            )

    def step(self, current, rwd, future, done, act):
        self.memory.add_memory(current, rwd, future, done, act)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) >= self.mini_batch_size:
                self.train()

    def train(self):
        current_states, future_states, rewards, actions, dones = self.memory.get_sample(self.mini_batch_size)

        current_qs = self.model_train.predict_on_batch(current_states).numpy()
        future_qs = self.model_target.predict_on_batch(future_states).numpy()
        a = np.argmax(self.model_train.predict_on_batch(future_states).numpy(), axis=1)
        max_future_qs = future_qs[np.arange(self.mini_batch_size), a]

        new_qs = rewards + self.discount_rate * max_future_qs * (1 - dones)
        current_qs[np.arange(self.mini_batch_size), actions] = new_qs

        self.model_train.train_on_batch(
            current_states,
            current_qs,
        )

        self.soft_update_model_weights()

    def soft_update_model_weights(self):
        new_weights = [
            self.tau * train_para + (1.0 - self.tau) * predict_para
            for train_para, predict_para in zip(self.model_train.get_weights(), self.model_target.get_weights())
        ]

        self.model_target.set_weights(new_weights)


def ddqn():
    env = gym.make("LunarLander-v2")
    n_episode = 1000
    agent = Agent(len(env.observation_space.high), env.action_space.n)
    history = list()

    for current_episode in range(n_episode):
        current_state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(current_state)
            future_state, reward, done, _ = env.step(action)
            agent.step(current_state, reward, future_state, done, action)

            episode_reward += reward
            current_state = future_state

        agent.decay_epsilon()

        episode_reward = round(episode_reward, 4)
        history.append(episode_reward)
        print(current_episode, episode_reward)

        if len(history) >= 100:
            if np.mean(history[-100:]) > 200:
                break

    env.close()

    path = "models/ddqn_model"
    if not os.path.isdir(path):
        os.mkdir(path)
    agent.model_train.save(path)

    plt.plot(np.arange(len(history)), history)
    plt.ylabel("Score")
    plt.xlabel("")
    plt.show()


def show_result():
    agent = load_model("models/ddqn_model")
    env = gym.make("LunarLander-v2")

    for i in range(10):
        done = False
        episode_reward = 0
        state = env.reset()
        while not done:
            env.render()
            action = np.argmax(
                agent.predict(np.array([state]))[0]
            )
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        print(i, round(episode_reward, 4))


if __name__ == "__main__":
    ddqn()
    # show_result()
