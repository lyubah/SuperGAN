"""
Main file where generator training and metric calculations take place.
"""
from argparse import Namespace, ArgumentParser
from typing import Tuple

from colorama import Fore
from numpy import ndarray

import config_file_parser
import saving_module
from gan_model import GanModel


def parse_command_line_args() -> Namespace:
    """
    Parses command line arguments

    :return:
    """
    parser = ArgumentParser(description='''
                                                        SuperGAN utilizes a Generative Adversarial
                                                        Network (GAN) to produce synthetic data 
                                                        similar to real data.
                                                        ''',
                            epilog='Thanks for using our program. Cheers!')
    parser.add_argument('-s', '--save', action='store_true', help='Save the current state of a trained GAN')
    parser.add_argument('-S', '--save_samples', action='store_true', help='Save samples of data')
    parser.add_argument('-l', '--load', action='store_true',
                        help='Load a pre-trained GAN model, and generate a number of sample')
    parser.add_argument('-c', '--count', type=int, help='The number of samples to generate', default=5)
    parser.add_argument('config', type=str, help='The .toml configuration file that needs to be loaded')
    return parser.parse_args()


def generate_data_samples(arguments: Namespace, gan_model: GanModel):
    # save a given number of samples
    for i in range(arguments.count):
        # compute the performance metrics
        synthetic_data, generator_classifier_accuracy = gan_model.generate_synthetic_data()
        saving_module.save_data_sample(synthetic_data, i + 1, gan_model.class_label)


def compute_performance_metrics(gan_model: GanModel) -> \
        Tuple[ndarray, ndarray, ndarray, float]:
    # the user feels like a hacker, but we
    # have to remind them that they are a 90's hacker

    # GENERATE SYNTHETIC DATA AND GET CLASSIFIER ACCURACY
    synthetic_data, generator_classifier_accuracy = gan_model.generate_synthetic_data()
    print(f'Classifier accuracy for synthetic data: {generator_classifier_accuracy}')

    # COMPUTE RTS AND STS METRICS
    mean_RTS_sim, mean_STS_sim = gan_model.compute_rts_sts(synthetic_data)
    print(f'RTS similarity: {mean_RTS_sim}')
    print(f'STS similarity: {mean_STS_sim}')

    SFD = gan_model.compute_statistical_feature_distance(syn_data=synthetic_data)
    print(f'Statistical Feature Distance (SFD): {SFD}')

    # not entirely sure why this is being computed, but maybe its important
    one_segment_real = gan_model.compute_one_segment_real()

    return synthetic_data, mean_RTS_sim, mean_STS_sim, generator_classifier_accuracy


def train_model(arguments: Namespace, gan_model: GanModel):
    # set the generator accuracy and step
    generator_classifier_accuracy = 0
    epoch = 1
    while generator_classifier_accuracy < gan_model.training_parameters.accuracy_threshold:
        # make the wrapper green, so that
        # the user feels like an elite hacker
        print(Fore.GREEN)
        epoch_string = f'------------------------------Epoch: {epoch}------------------------------'
        print(epoch_string)

        print(Fore.MAGENTA)

        # TRAIN DISCRIMINATOR AND GENERATOR AND DISPLAY ACCURACY FOR EACH
        discriminator_acc, gen_discriminator_acc = gan_model.train_discriminator()
        print(f'Discriminator accuracy (D ACC): {discriminator_acc}')
        print(f'Generator accuracy in tricking the discriminator: {gen_discriminator_acc}')

        # compute performance metrics
        synthetic_data, mean_RTS_sim, mean_STS_sim, generator_classifier_accuracy = compute_performance_metrics(gan_model)

        # continue the aforesaid sorcery
        print(Fore.GREEN)
        print('-' * len(epoch_string))

        # write the training results to a csv, note that it does this in
        # append mode
        if gan_model.write_train_results:
            gan_model.write_training_results(current_epoch=epoch,
                                             discriminator_accuracy=discriminator_acc,
                                             generator_discriminator_acc=gen_discriminator_acc,
                                             generator_classifier_acc=generator_classifier_accuracy,
                                             mean_rts_similarity=mean_RTS_sim,
                                             mean_sts_similarity=mean_STS_sim)

        epoch += 1

    if gan_model.request_save or arguments.save:
        gan_model.save_model_to_directory(current_epoch=epoch)

    # end the foolishness
    print(Fore.RESET)


def main():
    """
    Main method.
    """
    args = parse_command_line_args()

    # obtain relevant data from the .conf file and create GAN model
    training_parameters, weights, names, model_data = config_file_parser.ModelConfigParser().parse_config()
    gan_model = GanModel(training_parameters, weights, names, model_data, args.config)

    if args.load:
        compute_performance_metrics(gan_model)
    else:
        train_model(args, gan_model)

    if args.save_samples:
        generate_data_samples(args, gan_model)


if __name__ == '__main__':
    # go to the main method
    main()
