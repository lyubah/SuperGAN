# SuperGAN: A Supervised Generative Adversarial Framework for Synthetic Sensor Data Generation

## Overview

This repository contains code related to the SuperGAN (supervised generative adversarial network) framework. 

## How to run

#### Text file for submitting test cases:

There are two configuration files SuperGAN reads.

The first, `model.conf` provides hyperparameters used for training. It also provides save locations for the generator and discriminator
in `models.generator_filename` and `models.discriminator_filename`. This file should be in .toml format. An example is provided in this
repository.

The second is a .toml file to be provided via a command line parameter. This toml file should have at minimum the following data:
* `data_file_path` : The path to the dataset being trained on.
* `classifier_path` : The path to the pre-trained classifier.
* `class_label` : The class to generate. 

#### Format of data:

As our network generates time-series data, the data must be in the form (num_samples, seg_length, num_channels). Since
our network is supervised, the corresponding class labels are also required. Two versions must be included: 1) standard
integer encoding 2) one-hot vector encoding. It is assumed that the data as well as standard and one-hot labels are
included in a .h5 file and are named "X", "y" and "y_onehot" respectively.

#### Running from command line:

Format is `python main.py config_file.toml`

The following command line parameters are accepted:
* `-h, --help` : Display a help message
* `-s, --save` : Saves the Generator and Discriminator after training. This save location is provided in `model.conf`
* `-S, --save_samples` : Saves a number of generated samples of data after training. 
* `-l, --load` : Loads a pre-trained GAN to generate samples from. The file location is provided in `model.conf`
* `-C, --ignore_classifier` : Trains the GAN without the classifier loss function.
* `-R, --ignore_regularizer` : Trains the gAN without the SFD regularization loss function.
* `-c COUNT, --count COUNT` : Specifies the number of samples to generate.

#### Dependencies:

Our program requires the following packages: `tensorflow, Keras, sklearn, numpy, matplotlib, h5py, toml`

## Files in the repository

`CASAS_adlnormal_dataset.h5`, `sports_data_accelerometer.h5`, and `sports_data_gyroscope.h5` : Datasets created using the preprocessing scripts
in the [Data Preprocessing](https://github.com/SuperGAN-Public/Data-Preprocessing) repo.

`compute_rtr_similarity.py` : A script for calculating RTR similarity over some dataset and class label.

`config_file_parser.py` : Module for processing the `model.conf` file.

`do_experiments.rb` : A Ruby script which was used to automate experiments used in the paper.
Trains a model over every class and dataset.

`example.toml`, `example_w_save.toml` : Example .toml inputs to be provided via command line parameters. 

`gan_model.py` : Module for constructing GAN model given configuration.

`input_module.py` : Contains necessary functions for processing the .toml input file and loading the appropriate data.

`LSTM_accelerometer.h5`, `LSTM_adlnormal.h5`, `LSTM_gyroscope.h5` : Pre-trained classifiers for the three datasets.

`main.py` : Main file which takes the conditions from the .toml file and trains a generator for that given case.

`model.conf` : Configuration containing hyperparameters for training.

`models.py:` : Contains necessary models used in SuperGAN framework.

`plotting_module.py` : Contains necessary functions for displaying and saving plots of both real and generated data

`saving_module.py` : Contains necessary functions for saving training results and generator weights.

`train_simple_lstm.py` : Contains code for training the LSTM classifiers.

`training_module.py` : Functions for training generator and assessing data. In particular, contains functions for training
generator and discriminator, generating synthetic data, and computing the similarity metrics.
