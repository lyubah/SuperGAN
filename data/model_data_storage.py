"""
Stores model data in immutable wrapper classes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingParameters:
    """
    Class for keeping track of training parameters.
    """
    latent_dimension: int
    epochs: int
    batch_size: int
    test_size: int
    real_synthetic_ratio: int
    synthetic_synthetic_ratio: int
    discriminator_learning_rate: float
    accuracy_threshold: float
    num_features: int


@dataclass(frozen=True)
class Weights:
    """
    Class for keeping track of weights.
    """
    discriminator_loss_weight: int
    classifier_loss_weight: int
    sfd_loss_weight: int


@dataclass(frozen=True)
class Names:
    """
    Class for keeping track of names.
    """
    classifier_name: str
