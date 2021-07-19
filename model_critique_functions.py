"""

Model critique functions.

"""
import numpy as np
import tensorflow as tf
from keras import backend
from tensorflow_model_remediation.common.types import TensorType
from tensorflow_model_remediation.min_diff.losses import MMDLoss
from typing import Optional, Any


def maximal_mean_discrepancy(input_tensor: TensorType,
                             output_tensor: TensorType,
                             predictions_transform: Any = tf.sigmoid,
                             sample_weight: Optional[TensorType] = None,
                             kernel='gaussian') -> int:
    """
    Computes the maximum mean discrepancy. Note that the
    lower the result, the more evidence that the distributions are the same.
    Tensorflow's code can be found here:

    https://github.com/tensorflow/model-remediation/blob/v0.1.5/tensorflow_model_remediation/min_diff/losses/mmd_loss.py#L27-L121

    :param input_tensor: The membership data, which is a tensor type
    (you probably want to input a tensor).
    :param output_tensor: The predictions, which is a tensor type. (you probably
    want to input a tensor)
    :param predictions_transform: The transformation function, we default to
    sigmoid.
    :param sample_weight: The sample weight.
    :param kernel: The kernel that is being used.
    :returns: The maximum mean discrepancy.

    """
    mmd_loss = MMDLoss(kernel=kernel,
                       predictions_transform=predictions_transform,
                       name='mmd_loss')
    return mmd_loss.call(membership=input_tensor,
                         predictions=output_tensor,
                         sample_weight=sample_weight)


def wasserstein_distance(y_true: TensorType,
                         y_pred: TensorType) -> float:
    """
    Computes the Wasserstein distance (or Earth Mover's Distance)
    between two 1 dimensional distributions. Uses code pulled from
    here:

    https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/

    :param y_true: The values observed as an input.
    :param y_pred: The values observed as an output.
    :returns: The computed distance between the distributions.
    """
    return backend.mean(y_true * y_pred)


def euc_dist_loss(actual_output_data: TensorType,
                  expected_output_data: TensorType) -> TensorType:
    """
    Utility function that computes the euclidean distance (SFD) loss
    within the Keras optimization function.

    :param actual_output_data: The actual output data as a Tensor.
    :param expected_output_data: The expected output data as a Tensor.
    :return: The euclidean distance loss as a Tensor.
    """
    return backend.sqrt(
        backend.sum(backend.square(actual_output_data - expected_output_data),
                    axis=-1))


def compute_statistical_feature_distance(real_features: np.ndarray,
                                         synthetic_features: np.ndarray) -> np.ndarray:
    """
    Utility function that computes the average statistical feature distance during training.

    :param real_features: A numpy array of real features.
    :param synthetic_features: A numpy array of synthetic features.
    :return: A numpy array the represents the statistical feature distance.
    """
    distance_vector: np.ndarray = np.sqrt(
        np.sum(np.square(real_features - synthetic_features), axis=1))
    SFD: np.ndarray = np.mean(distance_vector)
    return SFD
