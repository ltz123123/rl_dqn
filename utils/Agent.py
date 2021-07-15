from utils.Network import Network
from utils.PER import PER
from collections import deque
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf


class Agent:
    def __init__(
            self,
            env_space,
            action_space,
            batch_size=64,
            max_memory_len=2 ** 14,
            alpha=0.6,
            beta=0.4,
            n_step=3,
            update_every=200,
            n_atom=51,
            v_max=300.0,
            v_min=-400.0
    ):
        self.env_space = env_space
        self.action_space = action_space

        self.batch_size = batch_size
        self.max_memory_len = max_memory_len
        self.memory = PER(self.max_memory_len, env_space, alpha, beta)

        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)

        self.epsilon = 1
        self.epsilon_decay_rate = 0.995
        self.epsilon_min = 0.05

        self.lr = 0.001
        self.gamma = 0.99
        self.t_step = 0
        self.update_every = update_every

        self.n_atom = n_atom
        self.v_max = v_max
        self.v_min = v_min
        self.support_z = np.linspace(v_min, v_max, n_atom)
        self.delta_z = (v_max - v_min) / (n_atom - 1)

        self.model = Network(env_space, action_space, n_atom)
        self.optimizer = Adam(learning_rate=self.lr)
        self.model_target = Network(env_space, action_space, n_atom)
        self.update_target_model()

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon * self.epsilon_decay_rate,
            self.epsilon_min
        )

    def get_action(self, state, testing=False):
        if not testing:
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.action_space)
        q = np.sum(self.model(tf.expand_dims(state, 0), training=False) * self.support_z, axis=-1)
        return np.argmax(q)

    def step(self, current, rwd, future, done, act):
        loss = np.nan

        self.n_step_buffer.append([current, act, rwd, future, done])
        if len(self.n_step_buffer) == self.n_step:
            rwd, future_state, done = self.get_n_step_info()
            current, action = self.n_step_buffer[0][:2]
            self.memory.add_memory(current, rwd, future, done, act)

        if len(self.memory) >= self.batch_size:
            loss = self.train_model()
            # self.model.reset_noise()
            # self.model_target.reset_noise()

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            self.update_target_model()

        return loss

    def get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done

    def train_model(self):
        # sampling
        (
            tree_idx, is_weights, current_states, future_states, rewards, actions, dones
        ) = self.memory.get_sample(self.batch_size)
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        # loss calculation, model fitting
        next_action = np.argmax(
            np.sum(self.model(current_states, training=False) * self.support_z, axis=2), axis=-1
        )
        next_dist = self.model_target(future_states, training=False).numpy()  # [batch_size, n_action, n_atom]
        next_dist = next_dist[np.arange(self.batch_size), next_action]  # [batch_size, n_atom]

        t_z = rewards + (self.gamma ** self.n_step) * self.support_z * (1 - dones)  # [batch_size, n_atom]
        t_z = np.clip(t_z, self.v_min, self.v_max)
        b = (t_z - self.v_min) / self.delta_z

        l = np.floor(b)
        u = np.ceil(b)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.n_atom - 1) * (l == u))] += 1

        offset = np.expand_dims(np.linspace(0, (self.batch_size - 1) * self.n_atom, self.batch_size), 1)
        offset = np.broadcast_to(offset, (self.batch_size, self.n_atom))

        proj_dist = np.zeros_like(next_dist).reshape(-1)
        np.add.at(proj_dist, (l + offset).reshape(-1).astype(int), (next_dist * (u - b)).reshape(-1))
        np.add.at(proj_dist, (u + offset).reshape(-1).astype(int), (next_dist * (b - l)).reshape(-1))
        proj_dist = proj_dist.reshape(next_dist.shape)

        loss_idx = np.vstack([np.arange(self.batch_size), actions]).T
        loss_idx = tf.stop_gradient(loss_idx)
        is_weights = tf.stop_gradient(is_weights)
        with tf.GradientTape() as tape:
            log_p = self.model(current_states, log=True)
            log_p = tf.gather_nd(log_p, loss_idx)
            elementwise_loss = -tf.reduce_sum(proj_dist * log_p, axis=1)
            loss = tf.reduce_mean(elementwise_loss * is_weights)
            loss = tf.clip_by_norm(loss, 10.0)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # update priorities
        abs_error = np.abs(elementwise_loss) + self.memory.e
        clipped_abs_error = np.where(
            abs_error < self.memory.abs_error_upper_bound,
            abs_error,
            self.memory.abs_error_upper_bound
        )
        clipped_abs_error = np.power(clipped_abs_error, self.memory.alpha)
        for i in range(self.batch_size):
            self.memory.update_priority(tree_idx[i], clipped_abs_error[i])

        return loss

    def update_target_model(self):
        self.model_target.set_weights(self.model.get_weights())
