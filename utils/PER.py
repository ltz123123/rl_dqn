# ref
# https://github.com/jcborges/dqn-per/blob/master/Memory.py
#    ref of ref
#    https://github.com/rlcode/per/blob/master/SumTree.py
#    https://github.com/rlcode/per/blob/master/prioritized_memory.py
# https://github.com/Huixxi/TensorFlow2.0-for-Deep-Reinforcement-Learning/blob/d1c191e7bfbb44357a4066ced3b96fa8c847875a/07_noisynet.py#L205
import numpy as np


class PER:
    def __init__(self, n_leaf_node, obs_dim, alpha=0.6, beta=0.4):
        self.current_size = 0
        self.pointer = 0

        self.n_leaf_node = n_leaf_node
        self.n_total_node = 2 * n_leaf_node - 1
        self.tree = np.zeros(self.n_total_node, dtype=np.float32)
        self.tree[(self.n_leaf_node - 1) + self.pointer] = 1.0

        self.memory_current_state = np.zeros([n_leaf_node, obs_dim], dtype=np.float32)
        self.memory_future_state = np.zeros([n_leaf_node, obs_dim], dtype=np.float32)
        self.memory_reward = np.zeros([n_leaf_node], dtype=np.float32)
        self.memory_done = np.zeros([n_leaf_node], dtype=np.int32)
        self.memory_action = np.zeros([n_leaf_node], dtype=np.int32)

        self.e = 1e-6
        self.alpha = alpha
        self.beta = beta
        self.b_increment_per_sampling = 0.001
        self.abs_error_upper_bound = 1.0

    def add_memory(self, current, rwd, future, done, act):
        self.memory_current_state[self.pointer] = current
        self.memory_future_state[self.pointer] = future
        self.memory_reward[self.pointer] = rwd
        self.memory_done[self.pointer] = done
        self.memory_action[self.pointer] = int(act)

        self.update_priority(
            (self.n_leaf_node - 1) + self.pointer,
            self.max_priority ** self.alpha
        )

        self.pointer = (self.pointer + 1) % self.n_leaf_node
        if self.current_size < self.n_leaf_node:
            self.current_size += 1

    def update_priority(self, tree_idx, abs_error):
        priority_change = abs_error - self.tree[tree_idx]

        # propagate upward
        self.tree[tree_idx] += priority_change
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += priority_change

    def get_leaf(self, v):
        tree_idx = 0

        # downward search
        while True:
            left_node_idx = 2 * tree_idx + 1
            right_node_idx = left_node_idx + 1

            if left_node_idx >= self.n_total_node:
                break
            else:
                if v <= self.tree[left_node_idx]:
                    tree_idx = left_node_idx
                else:
                    v -= self.tree[left_node_idx]
                    tree_idx = right_node_idx

        return tree_idx

    def get_sample(self, mini_batch_size):
        # idx
        tree_idx_list = np.zeros(mini_batch_size, dtype=np.int)

        priority_segment = self.tree[0] / mini_batch_size
        for i in range(mini_batch_size):
            value = np.random.uniform(priority_segment * i, priority_segment * (i + 1))
            tree_idx_list[i] = self.get_leaf(value)

        memory_idx_list = tree_idx_list - (self.n_leaf_node - 1)

        # weight
        p_min = self.min_priority / self.tree[0]
        max_weight = (p_min * self.current_size) ** -self.beta

        p_sampling = self.tree[tree_idx_list] / self.tree[0]
        is_weights = np.power(p_sampling * self.current_size, -self.beta)
        is_weights /= max_weight

        self.beta = min(1.0, self.beta + self.b_increment_per_sampling)

        return (
            memory_idx_list + (self.n_leaf_node - 1),
            is_weights,
            self.memory_current_state[memory_idx_list],
            self.memory_future_state[memory_idx_list],
            self.memory_reward[memory_idx_list],
            self.memory_action[memory_idx_list],
            self.memory_done[memory_idx_list]
        )

    @property
    def min_priority(self):
        if self.current_size == self.n_leaf_node:
            return np.min(self.tree[-self.n_leaf_node:])
        else:
            return np.min(self.tree[-self.n_leaf_node:-self.n_leaf_node+self.current_size])

    @property
    def max_priority(self):
        return np.max(self.tree[-self.n_leaf_node:])

    def __len__(self):
        return self.current_size
