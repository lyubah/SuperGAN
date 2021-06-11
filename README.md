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

`accuracy_epochs.rb` : A script which is run after `do_experiments.rb` to obtain the average accuracy and number of training epochs for each
configuration of neither loss function, regularizer only, classifier only, and both loss functions.

`CASAS_adlnormal_dataset.h5`, `sports_data_accelerometer.h5`, and `sports_data_gyroscope.h5` : Datasets created using the preprocessing scripts
in the [Data Preprocessing](https://github.com/SuperGAN-Public/Data-Preprocessing) repo.

`compute_rtr_similarity.py` : A script for calculating RTR similarity over some dataset and class label.

`config_file_parser.py` : Module for processing the `model.conf` file.

`do_experiments.rb` : A Ruby script which was used to automate experiments used in the paper.
Trains a model over every class and dataset.

`example.toml`, `example_w_save.toml` : Example .toml inputs to be provided via command line parameters. 

`extract_labels.rb` : A script which is run after `do_experiments.rb` to obtain a CSV with all STS similarity scores presented
in our table of data. 

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

## Tables of Data:

To produce the data presented in these tables, first run `ruby do_experiments.rb`, then, for each dataset, run
`ruby extract_labels.rb dataset_name` (dataset names are `accelerometer`, `adlnormal`, and `gyroscope`) to obtain
a CSV file containing all STS similarity values. The RTR similarity values can be obtained using the `compute_rtr_similarity.py`
script, which is run as follows: `python3 compute_rtr_similarity.py -c class_label dataset_file.h5`

STS Values for Accelerometer Dataset:

|     | 0      | 1      | 2      | 3      | 4      | 5      | 6      | 7       | 8      |
|-----|--------|--------|--------|--------|--------|--------|--------|---------|--------|
|     | 0.7612 | 0.7435 | 0.9997 | 0.9965 | 0.9993 | 0.9958 | 0.9995 | -0.0710 | 0.9959 |
| R   | 0.9424 | 0.9631 | 0.9389 | 0.9783 | 0.9562 | 0.9574 | 0.9715 | 0.9449  | 0.0173 |
| C   | 0.2175 | 0.2736 | 0.0873 | 0.0241 | 0.2351 | 0.2930 | 0.0362 | 0.0334  | 0.2286 |
| CR  | 0.1572 | 0.2046 | 0.1506 | 0.0454 | 0.0908 | 0.1941 | 0.1665 | 0.0960  | 0.0610 |
| RTR | 0.0063 | 0.0075 | 0.0022 | 0.0014 | 0.0031 | 0.0016 | 0.0015 | 0.0019  | 0.0021 |

STS Values for Gyroscope Dataset:

|     | 0      | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8       |
|-----|--------|--------|--------|--------|--------|--------|--------|--------|---------|
|     | 0.9935 | 0.9950 | 0.8631 | 0.9985 | 0.8625 | 0.9974 | 0.7108 | 0.9961 | 0.9960  |
| R   | 0.9524 | 0.9679 | 0.9325 | 0.9955 | 0.9940 | 0.0161 | 0.9716 | 0.9931 | 0.8438  |
| C   | 0.0937 | 0.0496 | 0.0250 | 0.3785 | 0.1878 | 0.0327 | 0.2404 | 0.1271 | 0.0302  |
| CR  | 0.0267 | 0.4181 | 0.1065 | 0.4279 | 0.1306 | 0.1195 | 0.1001 | 0.2564 | -0.0195 |
| RTR | 0.6610 | 0.7779 | 0.3019 | 0.8334 | 0.8262 | 0.6446 | 0.7801 | 0.8865 | 0.3569  |

STS Values for CASAS ADL Dataset:

|     | 0      | 1      | 2      | 3      | 4       |
|-----|--------|--------|--------|--------|---------|
|     | 0.9931 | 0.9819 | 0.9740 | 0.9692 | 0.9830  |
| R   | 0.1048 | 0.0551 | 0.5977 | 0.0014 | 0.1567  |
| C   | 0.0898 | 0.1060 | 0.1820 | 0.0698 | 0.2372  |
| CR  | 0.0965 | 0.1391 | 0.1382 | 0.0273 | -0.0671 |
| RTR | 0.0032 | 0.0164 | 0.1518 | 0.0023 | 0.0382  |
