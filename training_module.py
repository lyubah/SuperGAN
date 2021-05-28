"""
Functions for training generator and assessing data. In particular, contains functions for
training generator and discriminator, generating synthetic data, and computing the similarity metrics
"""
from typing import Tuple

import numpy as np
from keras import backend as keras_backend
from keras.engine.functional import Functional
from keras.utils.np_utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import Tensor


def euc_dist_loss(actual_output_data: Tensor, expected_output_data: Tensor) -> Tensor:
    """
    Utility function that computes the euclidean distance (SFD) loss
    within the Keras optimization function.

    :param actual_output_data: The actual output data as a Tensor.
    :param expected_output_data: The expected output data as a Tensor.
    :return: The euclidean distance loss as a Tensor.
    """
    return keras_backend.sqrt(keras_backend.sum(keras_backend.square(actual_output_data
                                                                     - expected_output_data), axis=-1))


def compute_statistical_feature_distance(real_features: np.ndarray, synthetic_features: np.ndarray) -> np.ndarray:
    """
    Utility function that computes the average statistical feature distance during training.

    :param real_features: A numpy array of real features.
    :param synthetic_features: A numpy array of synthetic features.
    :return: A numpy array the represents the statistical feature distance.
    """
    distance_vector: np.ndarray = np.sqrt(np.sum(np.square(real_features - synthetic_features), axis=1))
    SFD: np.ndarray = np.mean(distance_vector)
    return SFD


def generate_input_noise(batch_size: int, latent_dim: int, time_steps: int) -> np.ndarray:
    """
    Function that generates random input by sampling from a normal distribution, note that the input
    varies at each time-step.

    :param batch_size: The size of the batch.
    :param latent_dim: The latent dimension.
    :param time_steps: The time-steps.
    :return: Input noise.
    """
    return np.reshape(np.array(np.random.normal(0, 1, latent_dim * time_steps * batch_size)),
                      (batch_size, time_steps, latent_dim))


def generate_synthetic_data(size: int, generator: Functional, latent_dim: int, time_steps: int) -> np.ndarray:
    """
    A utility function for generating a synthetic data set.

    :param size: The size of the synthetic data.
    :param generator: The generator model.
    :param latent_dim: The latent dimensions.
    :param time_steps: The time-steps.
    :return: Synthetic data as a numpy array.
    """
    noise: np.ndarray = generate_input_noise(size, latent_dim, time_steps)
    synthetic_data: np.ndarray = generator.predict(noise)
    return synthetic_data


def train_generator(batch_size: int,
                    input_data: np.ndarray,
                    class_label: int,
                    actual_features: np.ndarray,
                    num_labels: int,
                    model: Functional,
                    latent_dim: int) -> list:
    """
    A utility function for training the generator based on both the discriminator
    and the classifier output.

    :param batch_size: The size of the batch.
    :param input_data: The input data as a numpy array.
    :param class_label: The class label.
    :param actual_features: The actual features denoted as a numpy array.
    :param num_labels: The number of labels.
    :param model: The model, which is a functional object, and is either a discriminator or a classifier.
    :param latent_dim: The latent dimension.
    :return: The loss as a list.
    """

    noise: np.ndarray = generate_input_noise(batch_size, latent_dim, input_data.shape[1])

    # labels related to whether data is real or synthetic
    real_synthetic_labels: np.ndarray = np.ones([batch_size, 1])

    # labels related to the class of the data
    class_labels: int = to_categorical([class_label] * batch_size,
                                       num_classes=num_labels)
    loss: list = model.train_on_batch(noise, [real_synthetic_labels, class_labels, actual_features])

    return loss


def train_discriminator(batch_size: int,
                        input_data: np.ndarray,
                        generator_model: Functional,
                        discriminator_model: Functional,
                        latent_dim: int) -> list:
    """
    A function for training the discriminator based on the generator input.

    :param batch_size: The batch size as an integer.
    :param input_data: The input data as a numpy array.
    :param generator_model: The generator model as a keras Functional object.
    :param discriminator_model: The discriminator model as a keras Functional object.
    :param latent_dim: The latent dimension fo the discriminator.
    :return: The loss as a list.
    """

    # generates the synthetic data
    noise: np.ndarray = generate_input_noise(batch_size, latent_dim, input_data.shape[1])
    synthetic_data: np.ndarray = generator_model.predict(noise)

    # selects a random batch of real data
    indices_toKeep: np.ndarray = np.random.choice(input_data.shape[0], batch_size, replace=False)
    real_data: np.ndarray = input_data[indices_toKeep]

    # makes the full input and labels and feeds the aforementioned into the network
    full_input: np.ndarray = np.concatenate((real_data, synthetic_data))
    real_synthetic_label: np.ndarray = np.ones([2 * batch_size, 1])
    real_synthetic_label[batch_size:, :] = 0

    # trains the discriminator and returns the loss
    loss: list = discriminator_model.train_on_batch(full_input, real_synthetic_label)
    return loss


def compute_similarity_metrics(synthetic_input_data: np.ndarray,
                               real_input_data: np.ndarray,
                               batch_size: int,
                               real_synthetic_ratio: int,
                               synthetic_synthetic_ratio: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for computing the mean rts and sts similarity, which is in turn used to help monitor
    the generator training.

    :param synthetic_input_data: The synthetic input data as a numpy array.
    :param real_input_data: The real input data as a numpy array.
    :param batch_size: The batch size.
    :param real_synthetic_ratio: The real-to-synthetic ratio.
    :param synthetic_synthetic_ratio: The synthetic-to-synthetic ratio.
    :return: The similarity metrics as a tuple of numpy arrays containing
    the mean rts similarity and the mean sts similarity in the following form
    (numpy array, numpy array)
    """

    # denote necessary features and information
    # regarding the data shape
    num_segments: int = len(real_input_data)
    seq_length: int = real_input_data.shape[1]
    num_channels: int = real_input_data.shape[2]
    RTS_sims: list = []
    STS_sims: list = []

    # reshape the data into two dimensions
    real_input_data: np.ndarray = real_input_data.reshape(num_segments, seq_length * num_channels)
    synthetic_input_data: np.ndarray = synthetic_input_data.reshape(batch_size, seq_length * num_channels)

    # for each fake segment, calculate its similarity to a user defined number of real segments
    for i in range(batch_size):
        indices_toCompare: np.ndarray = np.random.choice(num_segments, real_synthetic_ratio, replace=False)
        for j in indices_toCompare:
            sim: np.ndarray = cosine_similarity(
                real_input_data[j].reshape(1, -1), synthetic_input_data[i].reshape(1, -1))
            sim: np.float = sim[0, 0]
            RTS_sims.append(sim)

    # gets the index of one random fake sample
    index: np.ndarray = np.random.choice(len(synthetic_input_data), 1)

    # computes the similarity between the user defined number
    # of synthetic segments to test for generator collapse.
    chosen_synthetic_value = synthetic_input_data[index]

    # remove this sample so we dont compare to itself
    synthetic_input_data = np.delete(synthetic_input_data, index, axis=0)
    synthetic_toCompare = synthetic_input_data[np.random.choice(len(synthetic_input_data), synthetic_synthetic_ratio)]

    for other_synthetic in synthetic_toCompare:
        sim2: np.ndarray = cosine_similarity(chosen_synthetic_value, other_synthetic.reshape(1, -1))
        sim2: np.float = sim2[0, 0]
        STS_sims.append(sim2)

    RTS_sims: np.ndarray = np.array(RTS_sims)
    STS_sims: np.ndarray = np.array(STS_sims)
    mean_RTS_sim = np.mean(RTS_sims)
    mean_STS_sim = np.mean(STS_sims)

    return mean_RTS_sim, mean_STS_sim
