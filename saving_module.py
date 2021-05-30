"""
Contains functions necessary for saving training results and generator weights.
"""

import os

import h5py
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

    save_keras_model(model, save_directory, filename)


def save_discriminator_data(model: Functional, epoch: int, class_label: int, save_directory: str) -> None:
    """
    Saves the discriminator data from a given keras model.


    :param class_label: The class label.
    :param model: A keras model that contains the discriminator.
    :param epoch: The current "iteration" that the model's generated data corresponds to.
    :param save_directory: The directory that all of this data should be saved.
    :return: Nothing, since this is a void function.
    """
    filename = f'D_epoch{epoch}_label_class{class_label}.h5'

    save_keras_model(model, save_directory, filename)


def save_classifier_data(model: Functional, epoch: int, save_directory: str) -> None:
    """
        Saves the classifier data from a given keras model.


        :param model: A keras model that contains the discriminator.
        :param epoch: The current "iteration" that the model's generated data corresponds to.
        :param save_directory: The directory that all of this data should be saved.
        :return: Nothing, since this is a void function.
        """
    filename = f'C_epoch{epoch}.h5'

    save_keras_model(model, save_directory, filename)


def save_keras_model(model: Functional, save_directory: str, filename: str) -> None:
    """
    Save a keras model to a given directory

    :param model: A keras model to be saved.
    :param save_directory: The directory that all of this data should be saved.
    :param filename: The name of a file.
    :return: Nothing.
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    filepath = os.path.join(save_directory, filename)

    # Zero: There is a warning here (not an error, saving works) and it is weird,
    # having trouble figuring it out!
    model.save(filepath)


def save_data_sample(data: np.ndarray, iteration: int, class_label: int) -> None:
    """
    Saves data samples.

    :param data: The data to be saved.
    :param iteration: The iteration of the data save.
    :param class_label: The class label, indicates the filtered value.
    :return: Nothing, void function.
    """
    folder_name = 'synthetic_samples'

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    filename = f'data_sample{iteration}_class{class_label}.h5'
    filepath = os.path.join(folder_name, filename)

    with h5py.File(filepath, 'w') as data_saver:
        if not os.path.exists('synthetic_samples'):
            os.mkdir('synthetic_samples')
        data_saver.create_dataset('X', data=data)


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

    # make sure that we aren't appending to the last one
    if epoch == 1 and os.path.exists(filename):
        os.remove(filename)

    header = 'Epoch,Disc_acc,GenDisc_acc,GenClass_acc,mean_RTS_sim,mean_STS_sim\n'
    to_write = f'{epoch},{discriminator_accuracy},{generator_discriminator_accuracy},{generator_class_accuracy},' \
               f'{mean_rts_sim},{mean_sts_sim}\n'
    with open(filename, mode='a', encoding='utf-8') as f:
        if epoch == 1:  # this helps to separate multiple results if the code is run multiple times
            f.write(header)
        f.write(to_write)
