from utils.NoisyLayer import NoisyDense
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, ReLU, BatchNormalization
from tensorflow.keras.models import Sequential
import tensorflow as tf


class Network(tf.keras.Model):
    def __init__(self, input_shape, n_action, n_atom, is_noisy=True, is_img_input=True):
        super().__init__()
        self.n_action = n_action
        self.n_atom = n_atom

        if is_img_input:
            # [116, 116, 3]
            # self.feature_layer = Sequential([
            #     Conv2D(32, 8, strides=4, padding="valid", activation="relu", input_shape=input_shape),
            #     Conv2D(64, 4, strides=3, padding="valid", activation="relu"),
            #     Conv2D(64, 3, strides=1, padding="valid", activation="relu"),
            #     Flatten()
            # ])
            # self.feature_output_shape = 3136

            # [75, 75, 3]
            self.feature_layer = Sequential([
                Conv2D(32, 5, strides=5, padding="valid", activation="relu", input_shape=input_shape),
                Conv2D(64, 5, strides=5, padding="valid", activation="relu"),
                Flatten()
            ])
            self.feature_output_shape = 576
        else:
            self.feature_layer = Sequential([
                Dense(128, activation="relu", kernel_initializer="he_uniform", input_shape=input_shape),
                # Dense(128, activation="relu", kernel_initializer="he_uniform"),
            ])
            self.feature_output_shape = 128

        n_hidden_nodes = 512 if is_img_input else 128
        if is_noisy:
            self.v1 = NoisyDense(n_hidden_nodes, input_dim=self.feature_output_shape)
            self.v2 = NoisyDense(n_atom, input_dim=n_hidden_nodes)
            self.a1 = NoisyDense(n_hidden_nodes, input_dim=self.feature_output_shape)
            self.a2 = NoisyDense(n_action * n_atom, input_dim=n_hidden_nodes)
        else:
            self.v1 = Dense(n_hidden_nodes, kernel_initializer="he_uniform")
            self.v2 = Dense(n_atom, kernel_initializer="he_uniform")
            self.a1 = Dense(n_hidden_nodes, kernel_initializer="he_uniform")
            self.a2 = Dense(n_action * n_atom, kernel_initializer="he_uniform")

    def call(self, inputs, training=True, mask=None, log=False):
        x = self.feature_layer(inputs)

        v = ReLU()(self.v1(x, training=training))
        v = ReLU()(self.v2(v, training=training))
        v = tf.reshape(v, (-1, 1, self.n_atom))

        a = ReLU()(self.a1(x, training=training))
        a = ReLU()(self.a2(a, training=training))
        a = tf.reshape(a, (-1, self.n_action, self.n_atom))

        dist = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))

        if log:
            dist = tf.nn.log_softmax(dist, axis=-1)
        else:
            dist = tf.nn.softmax(dist, axis=-1)

        return dist

    def reset_noise(self):
        self.v1.reset_noise()
        self.v2.reset_noise()
        self.a1.reset_noise()
        self.a2.reset_noise()
