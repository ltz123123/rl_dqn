from utils.NoisyLayer import NoisyDense
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
import tensorflow as tf


class Network(tf.keras.Model):
    def __init__(self, input_shape, n_action, n_atom):
        super().__init__()
        self.n_action = n_action
        self.n_atom = n_atom
        self.feature_layer = Sequential([
            Conv2D(32, 3, activation="relu", padding="same", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, 3, activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(128, 3, activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
        ])

        self.v1 = Dense(512, activation="relu", kernel_initializer="he_uniform")
        self.v2 = Dense(n_atom, activation="linear", kernel_initializer="he_uniform")

        self.a1 = Dense(512, activation="relu", kernel_initializer="he_uniform")
        self.a2 = Dense(n_action * n_atom, activation="linear", kernel_initializer="he_uniform")
        # self.v_layer = NoisyDense(n_atom, input_dim=128)
        # self.a_layer = NoisyDense(n_action * n_atom, input_dim=128)

    def call(self, inputs, training=None, mask=None, log=False):
        x = self.feature_layer(inputs)
        x = Flatten()(x)

        v = self.v1(x)
        v = self.v2(v)
        v = tf.reshape(v, (-1, 1, self.n_atom))

        a = self.a1(x)
        a = self.a2(a)
        a = tf.reshape(a, (-1, self.n_action, self.n_atom))

        dist = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))

        if log:
            dist = tf.nn.log_softmax(dist, axis=-1)
        else:
            dist = tf.nn.softmax(dist, axis=-1)

        return dist

    def reset_noise(self):
        self.v_layer.reset_noise(128)
        self.a_layer.reset_noise(128)
