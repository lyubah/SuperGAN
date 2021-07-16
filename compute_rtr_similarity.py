"""
Utility program that computes the rtr similarity of a given dataset from a .h5 file.
"""

import h5py as h5reader
import argparse as arg_parser
import numpy as np
import os

from itertools import combinations
from argparse import Namespace
from sklearn.metrics.pairwise import cosine_similarity


class InvalidH5FileError(Exception):
    """
    Exception raised when something goes wrong with reading the .h5 file
    """

    def __init__(self, message: str = 'Something went wrong with parsing the .h5 file'):
        super().__init__(message)


def compute_real_to_real_similarity(input_data: np.ndarray) -> np.ndarray:
    """
    Computes metrics regarding the real to real similarity, or the average cosine similarity
    between all pairs of real segments. Using a similar methodology to do this as is seen in the
    compute_similarity_metrics function in training_module.py in the SuperGAN project on GitHub.

    :param input_data: The input data of the .h5 datafile.

    :return: The real to real similarity as a float, not a numpy array (even though the type annotation says so).
    """
    num_segments: int = len(input_data)
    seq_length: int = input_data.shape[1]
    num_channels: int = input_data.shape[2]

    # Generate all pairwise cosine similarities.
    rtr_sims: list = []
    reshaped: np.ndarray = input_data.reshape(num_segments, seq_length * num_channels)

    for i in range(num_segments):
        for j in range(num_segments):
            sim: np.ndarray = cosine_similarity(reshaped[j].reshape(1, -1), reshaped[i].reshape(1, -1))
            rtr_sims.append(sim[0, 0])

    # return the mean
    return np.mean(np.array(rtr_sims))


def file_exists(path: str) -> bool:
    """
    Determine whether or not the file exists
    """
    return os.path.exists(path)


def parse_cli_arguments() -> Namespace:
    """
    Utility function that parses command line arguments
    """
    parser = arg_parser.ArgumentParser(description='''
                                                   This is a small script that computes
                                                   the RTR similarity of a given dataset
                                                    ''')
    parser.add_argument('filename', type=str, help='The name of the .h5 dataset')
    parser.add_argument('--class-label', '-c', default=0, type=int, help='The class label')

    return parser.parse_args()


def read_h5_file(file_name: str, class_label: int) -> np.ndarray:
    """
    Reads the given .h5 file.

    :return: The RTR similarity
    """
    # verify that the file extension is indeed .h5
    if not file_name.endswith('.h5'):
        raise InvalidH5FileError('Invalid file extension, the file must be of type (.h5)')

    # verify that the file actually exists
    if not file_exists(file_name):
        raise FileNotFoundError(f'Could not find the file "{file_name}", please verify that it is present!')

    with h5reader.File(file_name, mode='r') as h5_file:
        h5_file_keys = h5_file.keys()

        # verify the correctness of the keys
        if 'X' not in h5_file_keys or 'y' not in h5_file_keys:
            raise InvalidH5FileError('The keys of this .h5 file are invalid!')

        # get the output data and the indices to keep
        output_data = np.array(h5_file.get('y'))
        indices_to_keep = np.where(output_data == class_label)

        # get the input data
        input_data = np.array(h5_file.get('X'))[indices_to_keep]
        return compute_real_to_real_similarity(input_data)


def main() -> None:
    """
    De-facto main method, created to better organize code.
    """
    cli_args: Namespace = parse_cli_arguments()
    file_name: str = cli_args.filename

    rtr_similarity = read_h5_file(file_name, cli_args.class_label)
    print(f'RTR similarity (real-to-real similarity): {rtr_similarity}')


if __name__ == '__main__':
    main()
