"""
Contains functions necessary for saving training results and generator weights.
"""

import os

import numpy as np
from keras.models import Functional


def save_generated_data(model: Functional, epoch: int, class_label: int, save_directory: str) -> None:
    """
    Saves the generated data from a given keras model.

    :param model: A keras model that contains generated data.
    :param epoch: The current "iteration" that the model's generated data corresponds to.
    :param class_label: The class label that is being used.
    :param save_directory: The directory that all of this data should be saved.
    :return: Nothing, since this is a void function.
    """
    filename = f'G_epoch{epoch}_label_class{class_label}.h5'
    filepath = os.path.join(save_directory, filename)

    # verify that the path actually exists, make it if it doesn't
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    model.save(filepath)


def write_results(epoch: int, class_label: int, discriminator_accuracy: float, generator_discriminator_accuracy: float,
                  generator_class_accuracy: float, mean_rts_sim: np.ndarray, mean_sts_sim: np.ndarray) -> None:
    """
    A function that writes training results.

    :param epoch: The current "iteration" that that the results are being written.
    :param class_label: The class label that is being used.
    :param discriminator_accuracy: The accuracy of the discriminator.
    :param generator_discriminator_accuracy: The accuracy of the generator.
    :param generator_class_accuracy: The accuracy of the generator class.
    :param mean_rts_sim: The mean rts similarity, I think this is actually a float but VSCode was complaining
    so in the future if typing for numpy gets better do change this to a 32-bit numpy float or a 64-bit numpy float.
    :param mean_sts_sim: The mean sts similarity, I think this is actually a float but VSCode was complaining
    so in the future if typing for numpy gets better do change this to a 32-bit numpy float or a 64-bit numpy float.
    :return: Nothing, since this is a void function.
    """
    filename = f'Results_label_class_{class_label}.csv'

    header = 'Epoch,Disc_acc,GenDisc_acc,GenClass_acc,mean_RTS_sim,mean_STS_sim\n'
    to_write = f'{epoch},{discriminator_accuracy},{generator_discriminator_accuracy},{generator_class_accuracy},' \
               f'{mean_rts_sim},{mean_sts_sim}\n'
    with open(filename, mode='a', encoding='utf-8') as f:
        if epoch == 1:  # this helps to separate multiple results if the code is run multiple times
            f.write(header)
        f.write(to_write)
