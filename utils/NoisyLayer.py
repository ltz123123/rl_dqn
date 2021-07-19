import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


class NoisyDense(Layer):
    def __init__(self, units, input_dim, std_init=0.5):
        super().__init__()
        self.units = units
        self.input_dim = input_dim
        self.std_init = std_init
        self.reset_noise(input_dim)

        mu_range = 1 / np.sqrt(input_dim)
        mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        sigma_initializer = tf.constant_initializer(self.std_init / np.sqrt(input_dim))

        self.weight_mu = tf.Variable(
            initial_value=mu_initializer(shape=(input_dim, units), dtype='float32'),
            trainable=True
        )
        self.weight_sigma = tf.Variable(
            initial_value=sigma_initializer(shape=(input_dim, units), dtype='float32'),
            trainable=True, name="weight_sigma"
        )
        self.bias_mu = tf.Variable(
            initial_value=mu_initializer(shape=(units,), dtype='float32'),
            trainable=True
        )
        self.bias_sigma = tf.Variable(
            initial_value=sigma_initializer(shape=(units,), dtype='float32'),
            trainable=True, name="bias_sigma"
        )

    def call(self, inputs, training=True, **kwargs):
        if training:
            self.kernel = self.weight_mu + self.weight_sigma * self.weights_eps
            self.bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            self.kernel = self.weight_mu
            self.bias = self.bias_mu
        return tf.matmul(inputs, self.kernel) + self.bias

    @staticmethod
    def _scale_noise(dim):
        noise = tf.random.normal([dim])
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    def reset_noise(self, input_shape=None):
        # eps_in = self._scale_noise(input_shape)
        eps_in = self._scale_noise(self.input_dim)
        eps_out = self._scale_noise(self.units)
        self.weights_eps = tf.multiply(tf.expand_dims(eps_in, 1), eps_out)
        self.bias_eps = eps_out
