# Reference
# https://github.com/sanketsans/openAIenv/blob/master/DQN/LunarLander/LunarLander_DQN.ipynb
# https://github.com/plopd/deep-reinforcement-learning/blob/master/dqn/dqn_agent.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as kb
import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
kb.clear_session()


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


def build_model(input_shape, output_shape):
    if type(input_shape) == int:
        input_shape = (input_shape,)

    model = Sequential([
        Dense(128, activation="relu", input_shape=input_shape, kernel_initializer="he_uniform"),
        Dense(128, activation="relu", kernel_initializer="he_uniform"),
        Dense(output_shape, activation="linear", kernel_initializer="he_uniform")
    ])

    return model


class Agent:
    def __init__(self, env_space, action_space):
        self.env_space = env_space
        self.action_space = action_space

        self.t_step = 0
        self.update_every = 200

        self.mini_batch_size = 64
        self.max_memory_len = 50000
        self.memory = ReplayMemory(self.max_memory_len, env_space)

        self.epsilon = 1
        self.epsilon_decay_rate = 0.995
        self.epsilon_min = 0.05

        self.lr = 0.0005
        self.discount_rate = 0.995
        self.tau = 0.001

        self.model_train = build_model(env_space, action_space)
        self.model_train.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))
        self.model_target = build_model(env_space, action_space)
        self.update_target_model()

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
                self.model_train(np.array([state]), training=False)[0]
            )

    def step(self, current, rwd, future, done, act):
        self.memory.add_memory(current, rwd, future, done, act)

        if len(self.memory) >= self.mini_batch_size:
            self.train()
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            self.soft_update_target_model()

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

    def soft_update_target_model(self):
        new_weights = [
            self.tau * train_para + (1.0 - self.tau) * predict_para
            for train_para, predict_para in zip(self.model_train.get_weights(), self.model_target.get_weights())
        ]

        self.model_target.set_weights(new_weights)

    def update_target_model(self):
        self.model_target.set_weights(self.model_train.get_weights())


def run():
    env = gym.make("LunarLander-v2")
    n_episode = 1000
    agent = Agent(len(env.observation_space.high), env.action_space.n)
    history = list()
    ma = list()

    for current_episode in range(n_episode):
        current_state = env.reset()
        episode_reward = 0
        t = 0
        done = False

        while not done:
            action = agent.get_action(current_state)
            future_state, reward, done, _ = env.step(action)
            agent.step(current_state, reward, future_state, done, action)

            episode_reward += reward
            current_state = future_state
            t += 1

        agent.decay_epsilon()

        episode_reward = round(episode_reward, 4)
        history.append(episode_reward)
        print(f"{current_episode: >4}: {t: >4}, {episode_reward}")

        if len(history) >= 100:
            ma.append(np.mean(history[-100:]))
            if ma[-1] >= 200:
                break

    env.close()

    plt.figure(figsize=(10, 7))
    plt.plot(np.arange(len(history)), history, linewidth=0.2)
    plt.plot(
        np.arange(len(ma)) + 99, ma
    )
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.grid(True)
    plt.savefig("1ddqn_fig.png")
    plt.show()


def show_result():
    agent = load_model("saved_models/ddqn_model")
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
    run()
    # show_result()
