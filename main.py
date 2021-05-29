"""
Main file where generator training and metric calculations take place.
"""

from colorama import Fore
from keras.models import load_model, Functional
from gan_model import GanModel
from argparse import Namespace

import config_file_parser
import argparse


def load_pretrained_model(filepath: str) -> Functional:
    """
    Loads a pre-trained model.

    :return: A trained keras model.
    """
    return load_model(filepath)


def parse_command_line_args() -> Namespace:
    """
    Parses command line arguments

    :return:
    """
    parser = argparse.ArgumentParser(description='''
                                                        SuperGAN utilizes a Generative Adversarial
                                                        Network (GAN) to produce synthetic data 
                                                        similar to real data.
                                                        ''',
                                     epilog='Thanks for using our program. Cheers!')
    parser.add_argument('-s', '--save', action='store_true', help='Save the current state of a trained GAN')
    parser.add_argument('-l', '--load', type=str, help='Load a pre-trained GAN model, and generate a number of sample')
    parser.add_argument('-c', '--count', type=int, help='The number of samples to generate', default=5)
    parser.add_argument('config', type=str, help='The .toml configuration file that needs to be loaded')
    return parser.parse_args()


def main():
    """
    Main method.
    """
    args = parse_command_line_args()

    # obtain relevant data from the .conf file
    config_data = config_file_parser.ModelConfigParser()
    training_parameters, weights, names, model_data = config_data.parse_config()
    gan_model = GanModel(training_parameters, weights, names, model_data, args.config)

    # initialize variables for later
    discriminator_acc = None
    gen_discriminator_acc = None
    mean_RTS_sim = None
    mean_STS_sim = None

    # set the generator accuracy and step
    generator_classifier_accuracy = 0
    epoch = 1
    while generator_classifier_accuracy < training_parameters.accuracy_threshold:
        # make the wrapper green, so that
        # the user feels like an elite hacker
        print(Fore.GREEN)

        epoch_string = f'------------------------------Epoch: {epoch}------------------------------'
        print(epoch_string)

        # the user feels like a hacker, but we
        # have to remind them that they are a 90's hacker
        print(Fore.MAGENTA)

        # TRAIN DISCRIMINATOR AND GENERATOR AND DISPLAY ACCURACY FOR EACH
        discriminator_acc, gen_discriminator_acc = gan_model.train_discriminator()
        print(f'Discriminator accuracy (D ACC): {discriminator_acc}')
        print(f'Generator accuracy in tricking the discriminator: {gen_discriminator_acc}')

        # GENERATE SYNTHETIC DATA AND GET CLASSIFIER ACCURACY
        synthetic_data, generator_classifier_accuracy = gan_model.generate_synthetic_data()
        print(f'Classifier accuracy for synthetic data: {generator_classifier_accuracy}')

        # COMPUTE RTS AND STS METRICS
        mean_RTS_sim, mean_STS_sim = gan_model.compute_rts_sts(synthetic_data)
        print(f'RTS similarity: {mean_RTS_sim}')
        print(f'STS similarity: {mean_STS_sim}')

        SFD = gan_model.compute_statistical_feature_distance(syn_data=synthetic_data)
        print(f'Statistical Feature Distance (SFD): {SFD}')

        # continue the aforesaid sorcery
        print(Fore.GREEN)

        # make a nice wrapper
        print('-' * len(epoch_string))
        print()

        epoch += 1

        # not entirely sure why this is being computed, but maybe its important
        one_segment_real = gan_model.compute_one_segment_real()

    # end the foolishness
    print(Fore.RESET)

    # make sure that we are not referencing before assignment
    if discriminator_acc is not None and gen_discriminator_acc is not None \
            and mean_STS_sim is not None and mean_RTS_sim is not None:
        gan_model.save_write_handler(current_epoch=epoch,
                                     discriminator_accuracy=discriminator_acc,
                                     generator_discriminator_acc=gen_discriminator_acc,
                                     generator_classifier_acc=generator_classifier_accuracy,
                                     mean_rts_similarity=mean_RTS_sim,
                                     mean_sts_similarity=mean_STS_sim,
                                     save_file=args.save)


if __name__ == '__main__':
    # go to the main method
    main()
