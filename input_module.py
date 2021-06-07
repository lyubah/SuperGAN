"""
Contains functions necessary for processing the .txt input file and loading the appropriate data
"""
from typing import Tuple

import h5py
import numpy as np
import toml


class InputModuleConfiguration:
    """
    Basically using this to store attributes
    """
    save_directory: str = 'models/'
    request_save: bool = False
    data_file_path: str = None
    classifier_path: str = None
    class_label: int = 0
    write_train_results: bool = False

    def __init__(self):
        pass

    def is_valid_config(self) -> bool:
        """
        Verifies that well, it does GAN things.
        :return: A boolean representing whether or not the TOML file is even usable
        """
        return self.data_file_path is not None and self.classifier_path is not None


def parse_input_file(file_name: str) -> InputModuleConfiguration:
    """
    Parses an input file of the given filename.
    :param file_name: The name of an input file.
    :return: A tuple of data that was parsed from a file.
    """
    if not file_name.endswith('.toml'):
        raise ValueError

    with open(file_name, mode='r', encoding='utf-8') as input_file:
        configuration_file = toml.load(input_file)
        input_module_config = InputModuleConfiguration()
        for key, value in configuration_file.items():
            setattr(input_module_config, key, value)

    if not input_module_config.is_valid_config():
        raise IOError

    return input_module_config


def load_data(filepath_data: str, class_label: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads data from an input file, based on a given filepath.

    :param filepath_data: The filepath that the .h5 file is located at.
    :param class_label: A class label, which is used for loading data from a selected class.
    :return: A 3-tuple of numpy array, formulated as follows (input_data, output_data, output_data_onehot)
    """

    def load_class_data(given_input_data: np.ndarray,
                        given_output_data: np.ndarray,
                        given_output_onehot: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        A helper function which loads data for a class which is specified by the class_label
        parameter of the surrounding function, filters out extraneous data.

        :param given_input_data: The given input data, which is a numpy array.
        :param given_output_data: The given output data, which is a numpy array.
        :param given_output_onehot: The given output onehot data, which is also a numpy array.
        :return: A three tuple, which contains the filtered data in the following form:
        (input_data, output_data, output_data_onehot)
        """

        # filter the output data
        indices_toKeep = np.where(given_output_data == class_label)

        # get the data at the indices to keep location
        in_data = given_input_data[indices_toKeep]
        out_data = given_output_data[indices_toKeep]
        out_data_onehot = given_output_onehot[indices_toKeep]
        return in_data, out_data, out_data_onehot

    with h5py.File(filepath_data, mode='r') as h5_file:
        h5_file_keys = h5_file.keys()

        # verify that the keys are all valid
        if 'X' not in h5_file_keys or 'y' not in h5_file_keys or 'y_onehot' not in h5_file_keys:
            raise IOError

        input_data = np.array(h5_file.get('X'))
        output_data = np.array(h5_file.get('y'))
        output_data_onehot = np.array(h5_file.get('y_onehot'))

        # close and prevent memory leaks
        h5_file.close()
        input_data, output_data, output_data_onehot = load_class_data(input_data,
                                                                      output_data,
                                                                      output_data_onehot)

    return input_data, output_data, output_data_onehot
