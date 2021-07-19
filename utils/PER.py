import numpy as np


class PER:
    def __init__(self, n_leaf_node, obs_shape, alpha=0.6, beta=0.4):
        self.current_size = 0
        self.pointer = 0

        self.n_leaf_node = n_leaf_node
        self.tree = np.zeros(n_leaf_node * 2 - 1)
        self.update_priority(n_leaf_node - 1, 1)  # initiate with value 1 for the first transition

        self.memory_current_state = np.zeros([n_leaf_node] + obs_shape, dtype=np.float32)
        self.memory_future_state = np.zeros([n_leaf_node] + obs_shape, dtype=np.float32)
        self.memory_reward = np.zeros([n_leaf_node], dtype=np.float32)
        self.memory_done = np.zeros([n_leaf_node], dtype=np.int32)
        self.memory_action = np.zeros([n_leaf_node], dtype=np.int32)

        self.e = 1e-3
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
            self.max_priority
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

            if left_node_idx >= len(self.tree):
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
            assert value <= self.tree[0]
            tree_idx_list[i] = self.get_leaf(value)

        memory_idx_list = tree_idx_list - (self.n_leaf_node - 1)

        p_sampling = self.tree[tree_idx_list] / self.tree[0]
        is_weights = np.power(p_sampling * self.current_size, -self.beta)
        is_weights /= np.max(is_weights)

        self.beta = min(1.0, self.beta + self.b_increment_per_sampling)

        return (
            memory_idx_list + (self.n_leaf_node - 1),
            is_weights.astype(np.float32),
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
