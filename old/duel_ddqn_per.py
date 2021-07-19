# ref
# https://github.com/jcborges/dqn-per/blob/master/Memory.py
#    ref of ref
#    https://github.com/rlcode/per/blob/master/SumTree.py
#    https://github.com/rlcode/per/blob/master/prioritized_memory.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gym
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.backend as kb
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from utils.PER import *


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
kb.clear_session()


def build_model(input_shape, output_shape):
    if type(input_shape) == int:
        input_shape = (input_shape,)

    inputs = Input(input_shape)

    x = Dense(128, activation="relu", kernel_initializer="he_uniform")(inputs)
    x = Dense(128, activation="relu", kernel_initializer="he_uniform")(x)

    v = Dense(1, kernel_initializer="he_uniform")(x)

    a = Dense(output_shape, kernel_initializer="he_uniform")(x)

    outputs = v + (a - kb.mean(a, axis=1, keepdims=True))

    model = Model(inputs=inputs, outputs=outputs)

    return model


class Agent:
    def __init__(self, env_space, action_space):
        self.env_space = env_space
        self.action_space = action_space

        self.mini_batch_size = 64
        self.max_memory_len = 2 ** 14
        self.memory = PER(self.max_memory_len, env_space)

        self.epsilon = 1
        self.epsilon_decay_rate = 0.995
        self.epsilon_min = 0.05

        self.lr = 0.0005
        self.discount_rate = 0.995
        self.tau = 0.001
        self.t_step = 0
        self.update_every = 4

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

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) >= self.mini_batch_size:
                self.train()
                self.soft_update_target_model()

    def train(self):
        (
            tree_idx, is_weights, current_states, future_states, rewards, actions, dones
        ) = self.memory.get_sample(self.mini_batch_size)
        temp_index_array = np.arange(self.mini_batch_size)

        current_qs = self.model_train.predict_on_batch(current_states).numpy()
        future_qs = self.model_target.predict_on_batch(future_states).numpy()

        a = np.argmax(self.model_train.predict_on_batch(future_states).numpy(), axis=1)
        max_future_qs = future_qs[temp_index_array, a]
        new_qs = rewards + self.discount_rate * max_future_qs * (1 - dones)

        target = current_qs.copy()
        target[temp_index_array, actions] = new_qs

        self.model_train.train_on_batch(
            current_states,
            target,
            sample_weight=is_weights
        )

        abs_error = np.abs((current_qs - target)[temp_index_array, actions]) + self.memory.e
        clipped_abs_error = np.where(
            abs_error < self.memory.abs_error_upper_bound,
            abs_error,
            self.memory.abs_error_upper_bound
        )
        clipped_abs_error = np.power(clipped_abs_error, self.memory.alpha)
        for i in range(self.mini_batch_size):
            self.memory.update_priority(tree_idx[i], clipped_abs_error[i])

    def update_target_model(self):
        self.model_target.set_weights(self.model_train.get_weights())

    def soft_update_target_model(self):
        new_weights = [
            (1.0 - self.tau) * predict_para + self.tau * train_para
            for train_para, predict_para in zip(self.model_train.get_weights(), self.model_target.get_weights())
        ]

        self.model_target.set_weights(new_weights)


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

            current_state = future_state
            episode_reward += reward
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

    path = "saved_models/duel_ddqn_per_model.h5"
    # agent.model_train.save(path)

    plt.figure(figsize=(10, 7))
    plt.plot(np.arange(len(history)), history, linewidth=0.2)
    plt.plot(
        np.arange(len(ma)) + 99, ma
    )
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.grid(True)
    plt.savefig("duel_ddqn_per_fig.png")
    plt.show()


def show_result():
    from PIL import Image
    agent = load_model("saved_models/duel_ddqn_per_model.h5")
    env = gym.make("LunarLander-v2")
    img_list = []
    for i in range(5):
        done = False
        episode_reward = 0
        state = env.reset()
        while not done:
            array = env.render(mode="rgb_array")
            img_list.append(array)
            action = np.argmax(
                agent.predict(np.array([state]))[0]
            )
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        print(i, round(episode_reward, 4))

    env.close()
    imgs = [Image.fromarray(im) for im in img_list]
    imgs[0].save("out.gif", save_all=True, append_images=imgs[1:], duration=60)


if __name__ == "__main__":
    run()
    # show_result()
    # m = build_model(8, 4, lr=0.005)
    # tf.keras.utils.plot_model(m, show_shapes=True)
