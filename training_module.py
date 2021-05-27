"""
Functions for training generator and assessing data. In particular, contains functions for
training generator and discriminator, generating synthetic data, and computing the similarity metrics
"""
from typing import Tuple

import numpy as np
from keras import backend as keras_backend
from keras.utils.np_utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from keras.engine.functional import Functional

from tensorflow import Tensor


# FUNCTION FOR COMPUTING EUCLIDEAN DISTANCE (SFD) LOSS WITHIN KERAS OPTIMIZATION FUNCTION
def euc_dist_loss(actual_output_data: Tensor, expected_output_data: Tensor) -> Tensor:
    return keras_backend.sqrt(keras_backend.sum(keras_backend.square(actual_output_data
                                                                     - expected_output_data), axis=-1))


# FUNCTION FOR COMPUTING AVERAGE STATISTICAL FEATURE DURING TRAINING
def compute_statistical_feature_distance(real_features: np.ndarray, synthetic_features: np.ndarray) -> np.ndarray:
    distance_vector: np.ndarray = np.sqrt(np.sum(np.square(real_features - synthetic_features), axis=1))
    SFD: np.ndarray = np.mean(distance_vector)
    return SFD


# FUNCTION FOR GENERATING RANDOM INPUT BY SAMPLING FROM NORMAL DISTRIBUTION (INPUT VARIES AT EACH TIME-STEP)
def generate_input_noise(batch_size: int, latent_dim: int, time_steps: int) -> np.ndarray:
    return np.reshape(np.array(np.random.normal(0, 1, latent_dim * time_steps * batch_size)),
                      (batch_size, time_steps, latent_dim))


# FUNCTION FOR GENERATING A SYNTHETIC DATA SET
def generate_synthetic_data(size: int, generator: Functional, latent_dim: int, time_steps: int) -> np.ndarray:
    noise: np.ndarray = generate_input_noise(size, latent_dim, time_steps)
    synthetic_data: np.ndarray = generator.predict(noise)
    return synthetic_data


# FUNCTION FOR TRAINING GENERATOR FROM BOTH DISCRIMINATOR AND CLASSIFIER OUTPUT
def train_generator(batch_size: int,
                    input_data: np.ndarray,
                    class_label: int,
                    actual_features: np.ndarray,
                    num_labels: int,
                    model: Functional,
                    latent_dim: int) -> list:
    noise: np.ndarray = generate_input_noise(batch_size, latent_dim, input_data.shape[1])

    # labels related to whether data is real or synthetic
    real_synthetic_labels: np.ndarray = np.ones([batch_size, 1])

    # labels related to the class of the data
    class_labels: int = to_categorical([class_label] * batch_size,
                                       num_classes=num_labels)
    loss: list = model.train_on_batch(noise, [real_synthetic_labels, class_labels, actual_features])

    return loss


# FUNCTION FOR TRAINING DISCRIMINATOR (FROM GENERATOR INPUT)
def train_discriminator(batch_size: int,
                        input_data: np.ndarray,
                        generator_model: Functional,
                        discriminator_model: Functional,
                        latent_dim: int) -> list:
    # GENERATE SYNTHETIC DATA
    noise: np.ndarray = generate_input_noise(batch_size, latent_dim, input_data.shape[1])
    synthetic_data: np.ndarray = generator_model.predict(noise)

    # SELECT A RANDOM BATCH OF REAL DATA
    indices_toKeep: np.ndarray = np.random.choice(input_data.shape[0], batch_size, replace=False)
    real_data: np.ndarray = input_data[indices_toKeep]

    # MAKE FULL INPUT AND LABELS FOR FEEDING INTO NETWORK
    full_input: np.ndarray = np.concatenate((real_data, synthetic_data))
    real_synthetic_label: np.ndarray = np.ones([2 * batch_size, 1])
    real_synthetic_label[batch_size:, :] = 0

    # TRAIN D AND RETURN LOSS
    loss: list = discriminator_model.train_on_batch(full_input, real_synthetic_label)
    return loss


# FUNCTION TO COMPUTE MEAN RTS AND STS SIMILARITY WHICH IS USED TO HELP MONITOR GENERATOR TRAINING
def compute_similarity_metrics(synthetic_input_data: np.ndarray,
                               real_input_data: np.ndarray,
                               batch_size: int,
                               real_synthetic_ratio: int,
                               synthetic_synthetic_ratio: int) -> Tuple[np.ndarray, np.ndarray]:
    # NECESSARY FEATURES REGARDING DATA SHAPE
    num_segments: int = len(real_input_data)
    seq_length: int = real_input_data.shape[1]
    num_channels: int = real_input_data.shape[2]
    RTS_sims: list = []
    STS_sims: list = []

    # RESHAPE DATA INTO 2 DIMENSIONS
    real_input_data: np.ndarray = real_input_data.reshape(num_segments, seq_length * num_channels)
    synthetic_input_data: np.ndarray = synthetic_input_data.reshape(batch_size, seq_length * num_channels)

    # FOR EACH FAKE SEGMENT, CALCULATE ITS SIMILARITY TO A USER DEFINED NUMBER OF REAL SEGMENTS
    for i in range(batch_size):
        indices_toCompare: np.ndarray = np.random.choice(num_segments, real_synthetic_ratio, replace=False)
        for j in indices_toCompare:
            sim: np.ndarray = cosine_similarity(real_input_data[j].reshape(1, -1), synthetic_input_data[i].reshape(1, -1))
            sim: np.float = sim[0, 0]
            RTS_sims.append(sim)

    # gets the index
    index: np.ndarray = np.random.choice(len(synthetic_input_data), 1)

    # ALSO COMPUTE SIMILARITY BETWEEN USER DEFINED NUMBER OF SYNTHETIC SEGMENTS TO TEST FOR GENERATOR COLLAPSE
    chosen_synthetic_value = synthetic_input_data[index]  # get one random fake sample
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
