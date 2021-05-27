"""
Main file where generator training and metric calculations take place.
"""

import sys
from typing import Tuple

import numpy as np
from colorama import Fore
from keras.engine.functional import Functional
from keras.models import Model, load_model, Functional
from numpy import ndarray
from sklearn.metrics import accuracy_score

import config_file_parser
import input_module
import models
import saving_module as save
import training_module as train
from data.model_data_storage import TrainingParameters, Weights, Names
from input_module import InputModuleConfiguration


class GanModel:
    feature_net: Functional
    discriminator_model: Functional
    discriminator: Functional
    generator: Functional
    num_classes: int
    input_shape: Tuple[int, int]
    num_channels: int
    seq_length: int
    num_seqs: int
    classifier: Functional
    write_train_results: bool
    request_save: bool
    model_save_directory: str
    class_label: int
    input_data: ndarray

    def __init__(self, training_param: TrainingParameters, weight: Weights, name: Names):
        """
        Constructs a new GAN model from the given training parameters, weights, and names.

        :param training_param: The training parameters.
        :param weight: The weights.
        :param name: The name.
        """
        self.training_parameters: TrainingParameters = training_param
        self.weights: Weights = weight
        self.names: Names = name

        # grab the file data and relevant information
        input_file_config: InputModuleConfiguration = input_module.parse_input_file(input_file)
        self.class_label = input_file_config.class_label
        self.model_save_directory = input_file_config.save_directory
        self.request_save = input_file_config.request_save

        y: ndarray
        y_onehot: ndarray
        self.input_data, y, y_onehot = input_module.load_data(input_file_config.data_file_path, self.class_label)
        self.write_train_results = False

        # load the pre-trained classifier
        self.classifier = load_model(input_file_config.classifier_path)
        self.classifier._name = self.names.classifier_name

        # set variables regarding the data shape
        self.num_seqs = self.input_data.shape[0]
        self.seq_length = self.input_data.shape[1]
        self.num_channels = self.input_data.shape[2]
        self.input_shape = (self.seq_length, self.num_channels)
        self.num_classes = y_onehot.shape[1]

        # create the generator
        self.generator = self._create_generator()

        # create the discriminator
        self.discriminator = self._create_discriminator()
        discriminator_to_freeze: Functional = self.discriminator
        self.discriminator_model = models \
            .compile_discriminator_model(discriminator=self.discriminator,
                                         learning_rate=training_param.discriminator_learning_rate)

        # create the statistical feature network and compute the feature vector for the real data
        # this is used in the loss function
        self.feature_net = self._create_feature_net()
        self.synthetic_data_train = self._train_synthetic_data()
        self.synthetic_data_test = self._test_generated_data()
        self._create_architecture(discriminator_to_freeze=discriminator_to_freeze)

    def _create_generator(self) -> Functional:
        """
        Creates a generator.

        :return:A generator as a Keras Functional object.
        """
        return models.create_generator(seq_length=self.seq_length,
                                       num_channels=self.num_channels,
                                       latent_dim=training_parameters.latent_dimension)

    def _create_discriminator(self) -> Functional:
        """
        Creates a discriminator.

        :return: A discriminator as a Keras Functional object.
        """
        return models.create_discriminator(seq_length=self.seq_length, num_channels=self.num_channels)

    def _create_feature_net(self) -> Functional:
        """
        Creates the statistical feature net.

        :return: A statistical feature net as a Keras Functional object.
        """
        return models.create_statistical_feature_net(seq_length=self.seq_length,
                                                     num_channels=self.num_channels,
                                                     num_features=self.training_parameters.num_features)

    def _train_synthetic_data(self) -> ndarray:
        """
        Trains synthetic data.

        :return: A numpy array of trained synthetic data.
        """
        return np.repeat(
            np.reshape(
                np.mean(
                    self.feature_net.predict(
                        self.input_data,
                        self.training_parameters.batch_size
                    ),
                    axis=0),
                (1, self.num_channels * self.training_parameters.num_features)),
            self.training_parameters.batch_size, axis=0)

    def _test_generated_data(self) -> ndarray:
        """
        Tests the generated data.

        :return: A numpy array of tested data.
        """
        return np.repeat(
            np.reshape(
                np.mean(
                    self.feature_net.predict(
                        self.input_data,
                        self.training_parameters.batch_size
                    ),
                    axis=0),
                (1, self.num_channels * self.training_parameters.num_features)),
            self.training_parameters.test_size, axis=0)

    def _create_architecture(self, discriminator_to_freeze: Functional) -> None:
        """
        Creates the full architecture where the output of the generator is fed to the
        discriminator and the classifier.

        :param discriminator_to_freeze: The Keras model discriminator that will be "frozen" (I hope).
        :return: Nothing, since this is conceptually a void function in a reasonably typed language.
        """
        for discriminator_layer in discriminator_to_freeze.layers:
            discriminator_layer.trainable = False

        self.GCD: Functional = Model(inputs=self.generator.input,
                                     outputs=[discriminator_to_freeze(self.generator.output),
                                              self.classifier(self.generator.output),
                                              self.feature_net(self.generator.output)])

        self.GCD.compile(loss={'D': 'binary_crossentropy', 'C': 'categorical_crossentropy', 'SFN': train.euc_dist_loss},
                         optimizer='adam', metrics={'D': 'accuracy', 'C': 'accuracy'},
                         loss_weights={'D': self.weights.discriminator_loss_weight,
                                       'C': self.weights.classifier_loss_weight,
                                       'SFN': self.weights.sfd_loss_weight})

    def train_discriminator(self) -> Tuple[np.float, np.float]:
        """
        Trains the discriminator. Mutates the discriminator and the generator.

        :return: The discriminator accuracy and the generator accuracy as a tuple in the following form
        (numpy array, numpy array).
        """
        discriminator_loss_vector: list = train \
            .train_discriminator(batch_size=self.training_parameters.batch_size,
                                 input_data=self.input_data,
                                 generator_model=self.generator,
                                 discriminator_model=self.discriminator_model,
                                 latent_dim=self.training_parameters.latent_dimension)

        GCD_loss_vec: list = train.train_generator(batch_size=self.training_parameters.batch_size,
                                                   input_data=self.input_data,
                                                   class_label=self.class_label,
                                                   actual_features=self.synthetic_data_train,
                                                   num_labels=self.num_classes,
                                                   model=self.GCD,
                                                   latent_dim=self.training_parameters.latent_dimension)

        # accuracy for the discriminator during its "turn" for training
        discriminator_accuracy: np.float = discriminator_loss_vector[1]

        # accuracy for the generator in tricking discriminator
        gen_accuracy: np.float = GCD_loss_vec[4]

        return discriminator_accuracy, gen_accuracy

    def generate_synthetic_data(self) -> Tuple[ndarray, float]:
        """
        Generates synthetic data.

        :return: Synthetic data that has been generated as a tuple containing the synthetic data
        and the accuracy of the generator class in the following form (numpy array, float).
        """
        syn_data: ndarray = train.generate_synthetic_data(size=self.training_parameters.test_size,
                                                          generator=self.generator,
                                                          latent_dim=self.training_parameters.latent_dimension,
                                                          time_steps=self.seq_length)
        pred: ndarray = np.argmax(self.classifier.predict(syn_data), axis=-1)
        true: list = [self.class_label] * self.training_parameters.test_size
        gen_class_acc: float = accuracy_score(true, pred)
        return syn_data, gen_class_acc

    def compute_rts_sts(self, syn_data: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Computes the similarity metrics.

        :param syn_data: The synthetic data as a numpy array.
        :return: A tuple of the following form (numpy array, numpy array) containing
        the rts similarity metrics and the sts similarity metrics.
        """
        return train.compute_similarity_metrics(synthetic_input_data=syn_data,
                                                real_input_data=self.input_data,
                                                batch_size=self.training_parameters.test_size,
                                                real_synthetic_ratio=self.training_parameters.real_synthetic_ratio,
                                                synthetic_synthetic_ratio=self.training_parameters.
                                                synthetic_synthetic_ratio)

    def compute_statistical_feature_distance(self, syn_data: ndarray) -> ndarray:
        """
        Computes the statistical feature distance.

        :param syn_data: The synthetic data.
        :return: The statistical feature distance as a numpy array.
        """
        synthetic_features = self.feature_net.predict(syn_data, self.training_parameters.test_size, verbose=0)
        return train.compute_statistical_feature_distance(synthetic_features, self.synthetic_data_test)

    def _save_model_to_directory(self, current_epoch: int) -> None:
        """
        Saves the model to a directory.

        :param current_epoch: The current epoch.
        :return: Nothing, since this function is a void function.
        """
        save.save_generated_data(self.generator, current_epoch, self.class_label, self.model_save_directory)

    def _write_train_results(self,
                             current_epoch: int,
                             discriminator_accuracy: np.float,
                             generator_discriminator_acc: np.float,
                             generator_classifier_acc: np.float,
                             mean_rts_similarity: ndarray,
                             mean_sts_similarity: ndarray) -> None:
        """
        Writes the training results.

        :param current_epoch: The current epoch.
        :param discriminator_accuracy: The accuracy of the discriminator.
        :param generator_discriminator_acc: The generator discriminator accuracy.
        :param generator_classifier_acc: The generator classifier accuracy.
        :param mean_rts_similarity: The mean rts similarity.
        :param mean_sts_similarity: The mean sts similarity.
        :return: Nothing, since, well, its a void function, I hope at least. Maybe its not,
        and then the code will break one day.
        """
        save.write_results(current_epoch,
                           self.class_label,
                           discriminator_accuracy,
                           generator_discriminator_acc,
                           generator_classifier_acc,
                           mean_rts_similarity,
                           mean_sts_similarity)

    def save_write_handler(self,
                           current_epoch: int,
                           discriminator_accuracy: np.float,
                           generator_discriminator_acc: np.float,
                           generator_classifier_acc: np.float,
                           mean_rts_similarity: ndarray,
                           mean_sts_similarity: ndarray) -> None:
        """
        This is basically a handler which determines whether to write the results to a file

        :param current_epoch: The current epoch.
        :param discriminator_accuracy: The accuracy of the discriminator.
        :param generator_discriminator_acc: The generator discriminator accuracy.
        :param generator_classifier_acc: The generator classifier accuracy.
        :param mean_rts_similarity: The mean rts similarity.
        :param mean_sts_similarity: The mean sts similarity.
        :return: Nothing.
        """
        # IF DESIRED, SAVE GENERATOR MODEL / WRITE TRAINING RESULTS
        if self.request_save:
            self._save_model_to_directory(current_epoch)
        if self.write_train_results:
            self._write_train_results(current_epoch,
                                      discriminator_accuracy,
                                      generator_discriminator_acc,
                                      generator_classifier_acc,
                                      mean_rts_similarity,
                                      mean_sts_similarity)

    def compute_one_segment_real(self) -> ndarray:
        """
        Computes the one segment real.

        :return: A numpy array that corresponds with the calculated one segment real.
        """
        return np.reshape(self.input_data[np.random.randint(0, self.input_data.shape[0], 1)],
                          (self.seq_length, self.num_channels))


def load_pretrained_model(filepath: str) -> Functional:
    """
    Loads a pre-trained model.

    :return: A trained keras model.
    """
    return load_model(filepath)


if __name__ == '__main__':
    # Load .toml file
    input_file = sys.argv[1]

    # obtain relevant data from the .conf file
    config_data = config_file_parser.ModelConfigParser()
    training_parameters, weights, names = config_data.parse_config()
    gan_model = GanModel(training_parameters, weights, names)

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

        gan_model.save_write_handler(current_epoch=epoch,
                                     discriminator_accuracy=discriminator_acc,
                                     generator_discriminator_acc=gen_discriminator_acc,
                                     generator_classifier_acc=generator_classifier_accuracy,
                                     mean_rts_similarity=mean_RTS_sim,
                                     mean_sts_similarity=mean_STS_sim)

        epoch += 1

        one_segment_real = gan_model.compute_one_segment_real()

    # end the foolishness
    print(Fore.RESET)
