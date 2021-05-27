"""
Contains models used in an EMBC paper. For further applications, users can
easily add any generator or discriminator architectures that they are interested in testing.
"""

from typing import Tuple

import numpy as np
import tensorflow as tf
from keras import backend as keras_backend
from keras.engine.keras_tensor import KerasTensor
from keras.layers import Dense, LSTM, Dropout, Input, Lambda
from keras.models import Model, Functional
from keras.optimizers import SGD
from keras.type.types import Layer
from tensorflow import Tensor


def create_discriminator(seq_length: int, num_channels: int) -> Functional:
    """
    Creates a discriminator architecture that was used in an EMBC paper.

    :param seq_length: The sequence length.
    :param num_channels: The number of channels.
    :return: A keras Functional object that represents the discriminator.
    """
    discriminator_input_shape: Tuple[int, int] = (seq_length, num_channels)
    discriminator_input: KerasTensor = Input(shape=discriminator_input_shape)
    discriminator: KerasTensor = Dropout(.5)(discriminator_input)
    discriminator: KerasTensor = LSTM(100, activation="tanh")(discriminator)
    discriminator: KerasTensor = Dense(1, activation="sigmoid")(discriminator)
    discriminator: Functional = Model(inputs=discriminator_input, outputs=discriminator, name="D")
    return discriminator


def create_generator(seq_length: int, num_channels: int, latent_dim: int) -> Functional:
    """
    Creates a generator architecture that was used in an EMBC paper.

    :param seq_length: The sequence length.
    :param num_channels: The number of channels.
    :param latent_dim: The number of latent dimensions.
    :return: A keras Functional object that represents the generator.
    """
    generator_input_shape: Tuple[int, int] = (seq_length, latent_dim)
    generator_input: KerasTensor = Input(shape=generator_input_shape)
    generator: KerasTensor = Dropout(.5)(generator_input)
    generator: KerasTensor = LSTM(128, return_sequences=True, activation="tanh")(generator)
    generator: KerasTensor = Dropout(.5)(generator)
    generator: KerasTensor = Dense(num_channels, activation="tanh")(generator)
    generator: Functional = Model(inputs=generator_input, outputs=generator)
    return generator


def create_statistical_feature_net(seq_length: int, num_channels: int, num_features: int) -> Functional:
    """
    Creates the full network for computing the statistical feature vector that
    is defined in the compute_stats function to make it flexible based on the sequence length
    and the number fo features without causing errors by passing non-tensor parameters.

    :param seq_length: The sequence length.
    :param num_channels: The number of channels.
    :param num_features: The number of features.
    :return: The statistical feature net.
    """

    def compute_stats(input_data: np.ndarray) -> Tensor:
        """
        Computes the stats.

        :param input_data: The input data which is represented as a numpy array.
        :return: A tensor object which contains data about the stats.
        """
        mean: Tensor = keras_backend.mean(input_data, axis=1, keepdims=True)
        standard_deviation: Tensor = keras_backend.std(input_data, axis=1, keepdims=True)
        variance: Tensor = keras_backend.var(input_data, axis=1, keepdims=True)

        # Zero: I think that this is where the error is, although I don't know for sure!
        # If it breaks I can just copy-paste code from an old commit and trial error until it runs
        x_max: Tensor = keras_backend.reshape(keras_backend.max(input_data, axis=1), (-1, 1, num_channels))
        x_min: Tensor = keras_backend.reshape(keras_backend.min(input_data, axis=1), (-1, 1, num_channels))
        p2p: Tensor = tf.subtract(x_max, x_min)
        amp: Tensor = tf.subtract(x_max, mean)
        rms: Tensor = keras_backend.reshape(
            keras_backend.sqrt(tf.reduce_sum(keras_backend.pow(input_data, 2), 1)), (-1, 1, num_channels))
        s2e: Tensor = keras_backend.reshape(
            tf.subtract(input_data[:, seq_length - 1, :], input_data[:, 0, :]), (-1, 1, num_channels))

        full_vec: Tensor = keras_backend.concatenate((mean, standard_deviation))
        full_vec: Tensor = keras_backend.concatenate((full_vec, variance))
        full_vec: Tensor = keras_backend.concatenate((full_vec, x_max))
        full_vec: Tensor = keras_backend.concatenate((full_vec, x_min))
        full_vec: Tensor = keras_backend.concatenate((full_vec, p2p))
        full_vec: Tensor = keras_backend.concatenate((full_vec, amp))
        full_vec: Tensor = keras_backend.concatenate((full_vec, rms))
        full_vec: Tensor = keras_backend.concatenate((full_vec, s2e))
        full_vec: Tensor = keras_backend.reshape(full_vec, (-1, num_features * num_channels))

        return full_vec

    def output_of_stat_layer(input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Gets the output of the stat layer.

        :param input_shape: The shape of the input.
        :return: The output of the stat layer as a tuple of integers (int, int).
        """
        return input_shape[0], input_shape[1] * num_features

    shape: Tuple[int, int] = (seq_length, num_channels)
    model_in: KerasTensor = Input(shape=shape)
    model_out: Layer = Lambda(compute_stats, output_shape=output_of_stat_layer)(model_in)

    model: Functional = Model(inputs=model_in, outputs=model_out, name="SFN")
    return model


# COMPILES DISCRIMINATOR MODEL TO TRAINED IN SUPERVISED GENERATIVE FRAMEWORK
def compile_discriminator_model(discriminator: Functional, learning_rate) -> Functional:
    """
    Compiles the discriminator model to be trained in a supervised generative framework.

    :param discriminator: The discriminator model.
    :param learning_rate: The learning rate
    :return: The mutated discriminator model.
    """
    optimizer: SGD = SGD(lr=learning_rate)
    model: Functional = Model(inputs=discriminator.input, outputs=discriminator.output)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model
