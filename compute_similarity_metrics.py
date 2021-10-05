"""
Utility program that computes the rtr similarity of a given dataset from a .h5 file.
"""

import argparse as arg_parser
import h5py as h5reader
import numpy as np
import os
from argparse import Namespace
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity


class InvalidH5FileError(Exception):
    """
    Exception raised when something goes wrong with reading the .h5 file
    """

    def __init__(self,
                 message: str = 'Something went wrong with parsing the .h5 file'):
        super().__init__(message)


def compute_real_to_real_similarity(real_input_data: np.ndarray,
                                    real_real_ratio: int) -> float:
    """
    Computes metrics regarding the real to real similarity, or the average
    cosine similarity between all pairs of real segments.

    :param real_input_data: The input data of the .h5 datafile.
    :param real_real_ratio: The ratio of real to real data to compare.

    :return: The real to real similarity as a float.
    """
    index: np.ndarray = np.random.choice(len(real_input_data), 1)

    # computes the similarity between the user defined number
    # of synthetic segments to test for generator collapse.
    chosen_real_value = real_input_data[index]

    # remove this sample so we don't compare to itself
    real_input_data = np.delete(real_input_data, index, axis=0)
    real_to_compare = real_input_data[
        np.random.choice(len(real_input_data), real_real_ratio)]

    # Generate all pairwise cosine similarities.
    rtr_sims: list = []

    for real in real_to_compare:
        sim2: np.ndarray = cosine_similarity(chosen_real_value,
                                             real.reshape(1, -1))
        sim2: np.float = sim2[0, 0]
        rtr_sims.append(sim2)

    # return the mean
    return np.mean(np.array(rtr_sims))


def compute_real_to_syn_similarity(real_input_data: np.ndarray,
                                   synthetic_input_data: np.ndarray,
                                   batch_size: int,
                                   real_synthetic_ratio: int) -> float:
    """
    Computes metrics regarding the real to synthetic similarity, or the average
    cosine similarity between chosen pairs of real and synthetic segments.

    :param real_input_data: Real data to compute RTS similarity
    :param synthetic_input_data: Synthetic data to compute RTS similarity.
    :param batch_size: The size of the batch.
    :param real_synthetic_ratio: The real to synthetic ratio.

    :return: The real to synthetic similarity as a float.
    """
    num_segments: int = len(real_input_data)
    seq_length: int = real_input_data.shape[1]
    num_channels: int = real_input_data.shape[2]

    # reshape the data into two dimensions
    real_input_data: np.ndarray \
        = real_input_data.reshape(num_segments, seq_length * num_channels)
    synthetic_input_data: np.ndarray \
        = synthetic_input_data.reshape(batch_size, seq_length * num_channels)

    # Generate all pairwise cosine similarities.
    rts_sims: list = []

    # for each fake segment, calculate its similarity
    # to a user defined number of real segments
    for i in range(batch_size):
        indices_to_compare: np.ndarray = np.random.choice(num_segments,
                                                          real_synthetic_ratio,
                                                          replace=False)
        for j in indices_to_compare:
            sim: np.ndarray = cosine_similarity(
                real_input_data[j].reshape(1, -1),
                synthetic_input_data[i].reshape(1, -1))
            sim: np.float = sim[0, 0]
            rts_sims.append(sim)

    # return the mean
    return np.mean(np.array(rts_sims))


def compute_syn_to_syn_similarity(synthetic_input_data: np.ndarray,
                                  synthetic_synthetic_ratio: float,
                                  batch_size,
                                  seq_length,
                                  num_channels) -> float:
    """
    Computes metrics regarding the synthetic to synthetic similarity,
    or the average cosine similarity between all pairs of synthetic segments.

    :param synthetic_input_data: Some synthetic input data.
    :param synthetic_synthetic_ratio: The ratio of synthetic to synthetic data.
    :param batch_size: The size of the batch.
    :param seq_length: The real data y length.
    :param num_channels: The real data z length.

    :return: The synthetic to synthetic similarity as a float.
    """
    # Generate all pairwise cosine similarities.
    sts_sims: list = []
    synthetic_input_data: np.ndarray \
        = synthetic_input_data.reshape(batch_size, seq_length * num_channels)

    # gets the index of one random fake sample
    index: np.ndarray = np.random.choice(len(synthetic_input_data), 1)

    # computes the similarity between the user defined number
    # of synthetic segments to test for generator collapse.
    chosen_synthetic_value = synthetic_input_data[index]

    # remove this sample so we don't compare to itself
    synthetic_input_data = np.delete(synthetic_input_data, index, axis=0)
    synthetic_to_compare = synthetic_input_data[
        np.random.choice(len(synthetic_input_data), synthetic_synthetic_ratio)]

    for other_synthetic in synthetic_to_compare:
        sim2: np.ndarray = cosine_similarity(chosen_synthetic_value,
                                             other_synthetic.reshape(1, -1))
        sim2: np.float = sim2[0, 0]
        sts_sims.append(sim2)

    # return the mean
    return np.mean(np.array(sts_sims))


def file_exists(path: str) -> bool:
    """
    Determine whether or not the file exists
    """
    return os.path.exists(path)


def parse_cli_arguments() -> Namespace:
    """
    Utility function that parses command line arguments
    """
    parser = arg_parser\
        .ArgumentParser(description='''
                                    This is a small script that computes
                                    the RTR similarity of a given dataset
                                    ''')
    parser.add_argument('filename', type=str,
                        help='The name of the .h5 dataset')
    parser.add_argument('--class-label', '-c', default=0, type=int,
                        help='The class label')

    return parser.parse_args()


def read_h5_file(file_name: str, class_label: int) -> float:
    """
    Reads the given .h5 file.

    :return: The RTR similarity
    """
    # verify that the file extension is indeed .h5
    if not file_name.endswith('.h5'):
        raise InvalidH5FileError(
            'Invalid file extension, the file must be of type (.h5)')

    # verify that the file actually exists
    if not file_exists(file_name):
        raise FileNotFoundError(
            f'Could not find the file "{file_name}"'
            f', please verify that it is present!')

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
        real_to_real_ratio = 10
        return compute_real_to_real_similarity(input_data, real_to_real_ratio)


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
