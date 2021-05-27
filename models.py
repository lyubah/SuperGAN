"""
Contains models used in EMBC paper. For further applications, users can
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


# CREATES DISCRIMINATOR ARCHITECTURE USED IN EMBC PAPER
from keras.type.types import Layer
from tensorflow import Tensor


def create_discriminator(seq_length: int, num_channels: int) -> Functional:
    discriminator_input_shape: Tuple[int, int] = (seq_length, num_channels)
    discriminator_input: KerasTensor = Input(shape=discriminator_input_shape)
    discriminator: KerasTensor = Dropout(.5)(discriminator_input)
    discriminator: KerasTensor = LSTM(100, activation="tanh")(discriminator)
    discriminator: KerasTensor = Dense(1, activation="sigmoid")(discriminator)
    discriminator: Functional = Model(inputs=discriminator_input, outputs=discriminator, name="D")
    return discriminator


# CREATES GENERATOR ARCHITECTURE USED IN EMBC PAPER
def create_generator(seq_length: int, num_channels: int, latent_dim: int) -> Functional:
    generator_input_shape: Tuple[int, int] = (seq_length, latent_dim)
    generator_input: KerasTensor = Input(shape=generator_input_shape)
    generator: KerasTensor = Dropout(.5)(generator_input)
    generator: KerasTensor = LSTM(128, return_sequences=True, activation="tanh")(generator)
    generator: KerasTensor = Dropout(.5)(generator)
    generator: KerasTensor = Dense(num_channels, activation="tanh")(generator)
    generator: Functional = Model(inputs=generator_input, outputs=generator)
    return generator


# CREATE FULL NETWORK FOR COMPUTING STATISTICAL FEATURE VECTOR
# DEFINED COMPUTE_STATS FUNCTION WITHIN HERE TO MAKE IT FLEXIBLE BASED ON SEQ LENGTH AND NUM FEATURES
# WITHOUT CAUSING ERRORS BY PASSING NON TENSOR PARAMETERS
def create_statistical_feature_net(seq_length: int, num_channels: int, num_features: int) -> Functional:
    def compute_stats(input_data: np.ndarray):
        mean: Tensor = keras_backend.mean(input_data, axis=1, keepdims=True)
        standard_deviation: Tensor = keras_backend.std(input_data, axis=1, keepdims=True)
        variance: Tensor = keras_backend.var(input_data, axis=1, keepdims=True)
        x_max: Tensor = keras_backend.reshape(keras_backend.max(input_data, axis=1), (-1, 1, 3))
        x_min: Tensor = keras_backend.reshape(keras_backend.min(input_data, axis=1), (-1, 1, 3))
        p2p: Tensor = tf.subtract(x_max, x_min)
        amp: Tensor = tf.subtract(x_max, mean)
        rms: Tensor = keras_backend.reshape(
            keras_backend.sqrt(tf.reduce_sum(keras_backend.pow(input_data, 2), 1)), (-1, 1, 3))
        s2e: Tensor = keras_backend.reshape(
            tf.subtract(input_data[:, seq_length - 1, :], input_data[:, 0, :]), (-1, 1, 3))

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
        return input_shape[0], input_shape[1] * num_features

    shape: Tuple[int, int] = (seq_length, num_channels)
    model_in: KerasTensor = Input(shape=shape)
    model_out: Layer = Lambda(compute_stats, output_shape=output_of_stat_layer)(model_in)

    model: Functional = Model(inputs=model_in, outputs=model_out, name="SFN")
    return model


# COMPILES DISCRIMINATOR MODEL TO TRAINED IN SUPERVISED GENERATIVE FRAMEWORK
def compile_discriminator_model(discriminator: Functional, learning_rate) -> Functional:
    optimizer: SGD = SGD(lr=learning_rate)
    model: Functional = Model(inputs=discriminator.input, outputs=discriminator.output)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model
