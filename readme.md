# SuperGAN: A Supervised Generative Adversarial Framework for Synthetic Sensor Data Generation

## Overview

This repository contains code related to the SuperGAN (supervised generative adversarial network) framework. This work
was done as part of the [Scalable Algorithms for Data Science Lab](https://scads.eecs.wsu.edu/) at Washington State
University.

## Key Details

#### Main idea:

To generate realistic synthetic labeled sensor data for remote health monitoring applications. Such data is valuable
because 

1) labelling data can be time-consuming, expensive and, depending on the domain, can require expert knowledge.

2) Sensor data often contains sensitive user information.

#### Supervised generative adversarial network:

The training of a generator depends on the feedback and training of a discriminator, and vice versa. However, balancing
the learning rates and training schedules for the two models is not an easy task. To increase training stability, our
generator also receives feedback from a highly-accurate pre-trained classifier. The intuition is that if an accurate
classifier can properly recognize the class label of the data being generated, the data is likely realistic.

#### Generating labelled data:

In order for the data to be useful for supervised learning tasks, it needs to be labeled. Our SuperGAN framework assists
us in generating labeled data because it "rewards" the generator for producing data that is similar enough to the given
class in order to be properly recognized. We then train a separate generator for each class. We found that this was
produced better results than conditioning over the labels as is done
with [Conditional Generative Adversarial Networks](https://arxiv.org/pdf/1411.1784.pdf)

#### Assessing the data:

Unlike images, time-series data can be difficult for a human to assess visually. Especially as the number of variables
increases. Thus, we also propose easily interpretable metrics for assessing the data. These metrics provide us with a
means of checking if 1) the generator is outputting a diverse set of data 2) the generator is ever simply copying the
training data 3) the generated data is similar to the real data. More details regarding the metrics are provided in
this [paper](https://scads.eecs.wsu.edu/wp-content/uploads/2018/04/embc_sensor_data_generation.pdf).

## How to run

#### Text file for submitting test cases:

Our program requires a text file containing the filepath to the data, the filepath to the pre-trained classifier, the
class label (in integer form) of the data to be generated, and the filepath for saving the generator weights (optional,
will not produce an error if not included). Each of these components must be on its own line. We provide an example of
this with the test.txt file.

#### Format of data:

As our network generates time-series data, the data must be in the form (num_samples, seg_length, num_channels). Since
our network is supervised, the corresponding class labels are also required. Two versions must be included: 1) standard
integer encoding 2) one-hot vector encoding. It is assumed that the data as well as standard and one-hot labels are
included in a .h5 file and are named "X", "y" and "y_onehot" respectively. The process of saving data in this format is
quite straightforward. More information is
provided [here](http://christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html).

#### Format of pre-trained classifier:

To allow for different classifier architectures to be utilized within the generation framework, we allow the user to
simply input the filepath to the classifier that they have pre-trained. We assume that this classifier is saved in a .h5
file. This is consistent with the instruction provided in
the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)

#### Running from command line:

Format is `python main.py test_case_filename.txt`

#### Dependencies:

Our program requires the following packages: `tensorflow, Keras, sklearn, numpy, matplotlib, h5py`

## Files in the repository

`example.txt:` example of necessary types of input to be included in test case file (not including a filepath for saving
the generator weights)

`example_w_save.txt:` same example as above except, in this case, the filepath for generator weights is included (will
automatically save weights after each epoch if this is included)

`LSTM_classifier.h5:` A pre-trained classifier with 1 LSTM layer with 100 units.

`ConvLSTM_classifier.h5:` A pre-trained classifier featuring five convolutional layers with 128, 96, 64, 48, 32 units,
respectively, (with max pooling after the 1st, 4th, and 5th convolutional layers) and two recurrent LSTM layers with 128
units each. More details regarding this architecture are provided [here](https://ieeexplore.ieee.org/document/8257960)

`sports_data_segmented.h5:` The normalized accelerometer data from the right leg (segmented into 5 second windows) from
the [Daily and Sports Activities Dataset](https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities). Note
that this only includes activities A10-A18.

`main.py:` main file which takes the conditions from the .txt file and trains a generator for that given case.

`input_module.py:` Contains necessary functions for processing the .txt input file and loading the appropriate data.

`saving_module.py:` Contains necessary functions for saving training results and generator weights.

`training_module.py:`Functions for training generator and assessing data. In particular, contains functions for training
generator and discriminator, generating synthetic data, and computing the similarity metrics.

`plotting_module.py:` Contains necessary functions for displaying and saving plots of both real and generated data (currently
written for tri-axial data but could be easily modified for applications with a different number of sensor channels)

`models.py:` Contains necessary models used in SuperGAN framework (for further applications, users can easily add any generator or
discriminator architectures that they are interested in testing).