from utils.NoisyLayer import NoisyDense
from tensorflow.keras.layers import Dense
import tensorflow as tf


class Network(tf.keras.Model):
    def __init__(self, input_shape, n_action, n_atom):
        super().__init__()
        self.n_action = n_action
        self.n_atom = n_atom
        self.fc1 = Dense(128, activation="relu", kernel_initializer="he_uniform", input_shape=(input_shape, ))
        self.fc2 = Dense(128, activation="relu", kernel_initializer="he_uniform")
        # self.v_layer = Dense(n_atom, activation="linear", kernel_initializer="he_uniform")
        # self.a_layer = Dense(n_action * n_atom, activation="linear", kernel_initializer="he_uniform")
        self.v_layer = NoisyDense(n_atom, input_dim=128)
        self.a_layer = NoisyDense(n_action * n_atom, input_dim=128)

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)

        v = self.v_layer(x)
        v = tf.reshape(v, (-1, 1, self.n_atom))

        a = self.a_layer(x)
        a = tf.reshape(a, (-1, self.n_action, self.n_atom))

        dist = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
        dist = tf.nn.softmax(dist, axis=-1)

        return dist

    def reset_noise(self):
        self.v_layer.reset_noise(128)
        self.a_layer.reset_noise(128)
