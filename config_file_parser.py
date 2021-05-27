"""
Utilities to parse the config file.
"""

import configparser
import os

from data.model_data_storage import Weights, TrainingParameters, Names


class ModelConfigParser:
    """
    A class that parses a configuration file
    for the project rather than utilizing hardcoded
    values. Makes the parameters easier to adjust over
    time.
    """

    def __init__(self):
        if not self.config_file_exists():
            self.create_default_config()

    @staticmethod
    def create_default_config():
        """
        Creates a default configuration file
        from a set of default values. Note that this
        is tailored to the original purpose of this
        program, to generate realistic fake health data.
        """
        model_maker = configparser.ConfigParser()
        model_maker['TRAINING_PARAMETERS'] = {
            'latent_dimension': '10',
            'epochs': '100',
            'batch_size': '25',
            'test_size': '100',
            'real_synthetic_ratio': '5',
            'synthetic_synthetic_ratio': '10',
            'discriminator_learning_rate': '0.01',
            'accuracy_threshold': '0.8',
            'num_features': '9'
        }
        model_maker['WEIGHTS'] = {
            'discriminator_loss_weight': '1',
            'classifier_loss_weight': '1',
            'sfd_loss_weight': '1'
        }
        model_maker['NAMES'] = {
            'classifier_name': 'C'
        }
        with open('model.conf', 'w') as configfile:
            model_maker.write(configfile)

    @staticmethod
    def config_file_exists(filepath: str = '.') -> bool:
        """
        Determines whether or not a configuration file exists
        in the given path, which defaults to the directory
        which this program was executed.

        :param filepath: A given filepath.
        :return: A boolean value representing whether the file exists.
        """
        return os.path.exists(os.path.join(filepath, 'model.conf'))

    @staticmethod
    def parse_config() -> (TrainingParameters, Weights, Names):
        """
        Parses the configuration file, and gets the relevant data.

        :return: A DataWrapper which stores all the necessary information:
        """

        def parse_training_parameters(key: configparser.SectionProxy) -> TrainingParameters:
            """
            Parses the training parameters in the provided key.

            :param key: A key that represents a map of training parameters.
            :return: A dataclass of parsed training parameters.
            """
            latent_dimension: int = int(key.get('latent_dimension', '10'))
            epochs: int = int(key.get('epochs', '100'))
            batch_size: int = int(key.get('batch_size', '25'))
            test_size: int = int(key.get('test_size', '100'))
            real_synthetic_ratio: int = int(key.get('real_synthetic_ratio', '5'))
            synthetic_synthetic_ratio: int = int(key.get('synthetic_synthetic_ratio', '10'))
            discriminator_learning_rate: float = float(key.get('discriminator_learning_rate', '0.01'))
            accuracy_threshold: float = float(key.get('accuracy_threshold', '0.8'))
            num_features: int = int(key.get('num_features', '9'))
            return TrainingParameters(latent_dimension=latent_dimension, epochs=epochs, batch_size=batch_size,
                                      test_size=test_size, real_synthetic_ratio=real_synthetic_ratio,
                                      synthetic_synthetic_ratio=synthetic_synthetic_ratio,
                                      discriminator_learning_rate=discriminator_learning_rate,
                                      accuracy_threshold=accuracy_threshold, num_features=num_features)

        def parse_weights(key: configparser.SectionProxy) -> Weights:
            """
            Parses the weights in the provided key.

            :param key: A key that represents a map of weights.
            :return: A dataclass of parsed weights.
            """
            discriminator_loss_weight: int = int(key.get('discriminator_loss_weight', '1'))
            classifier_loss_weight: int = int(key.get('classifier_loss_weight', '1'))
            sfd_loss_weight: int = int(key.get('sfd_loss_weight', '1'))
            return Weights(discriminator_loss_weight=discriminator_loss_weight,
                           classifier_loss_weight=classifier_loss_weight, sfd_loss_weight=sfd_loss_weight)

        def parse_names(key: configparser.SectionProxy) -> Names:
            """
            Parses the names in the provided key.

            :param key: A key that represents a map of names.
            :return: A dataclass of parsed names.
            """
            classifier_name: str = key['classifier_name']
            return Names(classifier_name=classifier_name)

        model_parser = configparser.ConfigParser()
        model_parser.read('model.conf')
        training_parameters: TrainingParameters = parse_training_parameters(model_parser['TRAINING_PARAMETERS'])
        weights: Weights = parse_weights(model_parser['WEIGHTS'])
        names: Names = parse_names(model_parser['NAMES'])
        return training_parameters, weights, names
