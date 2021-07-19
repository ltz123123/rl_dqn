import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as kb
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from utils.PER import PER
from utils.NoisyLayer import NoisyDense
from collections import deque


kb.clear_session()


class Network(tf.keras.Model):
    def __init__(self, input_shape, n_action, n_atom):
        super().__init__()
        self.n_action = n_action
        self.n_atom = n_atom
        self.fc1 = Dense(128, activation="relu", kernel_initializer="he_uniform", input_shape=(input_shape, ))
        self.fc2 = Dense(128, activation="relu", kernel_initializer="he_uniform")
        self.v_layer = NoisyDense(n_atom, 128)
        self.a_layer = NoisyDense(n_action * n_atom, 128)

        # self.v_layer = Dense(n_atom, activation="linear", kernel_initializer="he_uniform")
        # self.a_layer = Dense(n_action * n_atom, activation="linear", kernel_initializer="he_uniform")

    def call(self, inputs, training=None, mask=None, log=False):
        x = self.fc1(inputs)
        x = self.fc2(x)

        v = self.v_layer(x)
        v = tf.reshape(v, (-1, 1, self.n_atom))
        a = self.a_layer(x)
        a = tf.reshape(a, (-1, self.n_action, self.n_atom))

        dist = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
        if not log:
            dist = tf.nn.softmax(dist, axis=-1)
        else:
            dist = tf.nn.log_softmax(dist, axis=-1)

        return dist

    def reset_noise(self):
        self.a_layer.reset_noise()
        self.v_layer.reset_noise()


class Agent:
    def __init__(self, env_space, action_space, n_step=3, n_atom=15, v_max=300.0, v_min=-500.0):
        self.env_space = env_space
        self.action_space = action_space

        self.mini_batch_size = 64
        self.max_memory_len = 2 ** 14
        self.memory = PER(self.max_memory_len, env_space)

        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)

        self.epsilon = 1
        self.epsilon_decay_rate = 0.9
        self.epsilon_min = 0.05

        self.lr = 0.0001
        self.gamma = 0.99
        self.tau = 0.001
        self.t_step = 0
        self.update_every = 1000

        self.n_atom = n_atom
        self.v_max = v_max
        self.v_min = v_min
        self.support_z = np.linspace(v_min, v_max, n_atom)
        self.delta_z = (v_max - v_min) / (n_atom - 1)

        self.model_train = Network(env_space, action_space, n_atom)
        self.optimizer = Adam(learning_rate=self.lr)
        self.model_target = Network(env_space, action_space, n_atom)
        self.update_target_model()

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon * self.epsilon_decay_rate,
            self.epsilon_min
        )

    def get_action(self, state):
        # if np.random.rand() < self.epsilon:
        if np.random.rand() < 0:
            return np.random.randint(self.action_space)
        else:
            q = np.sum(self.model_train(np.array([state]), training=False) * self.support_z, axis=-1)
            return np.argmax(q)

    def step(self, current, rwd, future, done, act):
        self.n_step_buffer.append([current, act, rwd, future, done])
        if len(self.n_step_buffer) == self.n_step:
            rwd, future_state, done = self.get_n_step_info()
            current, action = self.n_step_buffer[0][:2]
            self.memory.add_memory(current, rwd, future, done, act)

        if len(self.memory) >= self.mini_batch_size:
            self.model_train.reset_noise()
            self.model_target.reset_noise()
            self.train()

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            self.update_target_model()

    def get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done

    def train(self):
        # sampling
        (
            tree_idx, is_weights, current_states, future_states, rewards, actions, dones
        ) = self.memory.get_sample(self.mini_batch_size)
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        # loss calculation, model fitting
        next_action = np.argmax(
            np.sum(self.model_train(current_states, training=False) * self.support_z, axis=2), axis=-1
        )
        next_dist = self.model_target(future_states, training=False).numpy()  # [batch_size, n_action, n_atom]
        next_dist = next_dist[np.arange(self.mini_batch_size), next_action]  # [batch_size, n_atom]

        t_z = rewards + (self.gamma ** self.n_step) * self.support_z * (1 - dones)  # [batch_size, n_atom]
        t_z = np.clip(t_z, self.v_min, self.v_max)
        b = (t_z - self.v_min) / self.delta_z

        l = np.floor(b)
        u = np.ceil(b)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.n_atom - 1) * (l == u))] += 1

        offset = np.expand_dims(np.linspace(0, (self.mini_batch_size - 1) * self.n_atom, self.mini_batch_size), 1)
        offset = np.broadcast_to(offset, (self.mini_batch_size, self.n_atom))

        proj_dist = np.zeros_like(next_dist).reshape(-1)
        np.add.at(proj_dist, (l + offset).reshape(-1).astype(int), (next_dist * (u - b)).reshape(-1))
        np.add.at(proj_dist, (u + offset).reshape(-1).astype(int), (next_dist * (b - l)).reshape(-1))
        proj_dist = proj_dist.reshape(next_dist.shape)

        loss_idx = np.vstack([np.arange(self.mini_batch_size), actions]).T
        loss_idx = tf.stop_gradient(loss_idx)
        is_weights = tf.stop_gradient(is_weights)
        with tf.GradientTape() as tape:
            log_p = self.model_train(current_states, log=True)
            log_p = tf.gather_nd(log_p, loss_idx)
            elementwise_loss = -tf.reduce_sum(proj_dist * log_p, axis=1)
            loss = tf.reduce_mean(elementwise_loss * is_weights)
            loss = tf.clip_by_norm(loss, 10.0)

        grads = tape.gradient(loss, self.model_train.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model_train.trainable_weights))

        # update priorities
        abs_error = np.abs(elementwise_loss) + self.memory.e
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
    agent = Agent([len(env.observation_space.high)], env.action_space.n)
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
    plt.savefig("figs/distributional_fig.png")
    plt.show()


if __name__ == "__main__":
    run()
    # show_result()
    # m = build_model(8, 4, lr=0.005)
    # tf.keras.utils.plot_model(m, show_shapes=True)
