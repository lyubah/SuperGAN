# SuperGAN

## Overview
This repository contains code related to the SuperGAN (supervised generative adversarial network) framework. Some information related to the project and its authors is currently anonymized to ensure the effectiveness of blind review processes. More information on the project will be provided at a later point.

## List of Files
* `ConvLSTM_classifier.h5` and `LSTM_classifier.h5` - These are pre-trained classifiers that can be loaded and used as "C" in the GAN's training.
* `example_w_save.txt` and `example.txt` - Config files that SuperGAN uses as input. These contain the following in order, each on a separate line:

   1. The input dataset (should be an .h5 filename)
   2. The classifier, C
   3. An integer representing a class label (for which class' data to generate)
   4. (Optionally) A directory for where to save the generators.

* `input_module.py` - Contains methods for loading the above config file
* `main.py` - The main program. Invokes all of the methods in the other files, and contains a loop to handle all of the training and computation of statistics.
* `models.py` - Contains code both for creating the G and D models, as well as computing statistics.
* `plotting_module.py` - Contains some code for plotting real and generated data. 
* `saving_module.py` - A module for saving the generator once it has been trained.
* `sports_data_segmented.h5` - The main dataset SuperGAN is trained on.
* `training_module.py` - Contains methods for training the GAN, as well as for computing some similarity metrics.