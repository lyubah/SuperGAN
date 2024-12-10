# -*- coding: utf-8 -*-
import argparse
import numpy as np
import statistics
from numpy.fft import fft
import torch.utils.data as dl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import math
import csv
from scipy import signal
from scipy.fft import fftshift
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy import signal
from numpy import savetxt
from collections import Counter
import itertools
import pickle
import os
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

# Load necessary Pytorch packages
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import matplotlib

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import numpy as np
from scipy.cluster.vq import kmeans, vq
#from TargetCnn import targetModel_cnn
from sklearn.model_selection import train_test_split
import sys
from random import randint
import copy
import numpy as np
np.random.seed(123)
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from tqdm import tqdm
device = torch.device('cpu')
from NN_functions import *
from utility_functions import *
from EENN_functions import * 

import time
def get_percentage_accuracy(actual_labels, output_labels):
    correct_predictions = (actual_labels == output_labels).sum()
    total_predictions = len(actual_labels)
    accuracy_percentage = (correct_predictions / total_predictions) * 100
    return accuracy_percentage
def main(args):
    
    args_dict = vars(args)
    exp_name = "-".join([str(args_dict[k]) for k in sorted(args_dict.keys())])
    
    output_path = os.path.join(args.dataset_name+"_outputs")
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    model_file_path = os.path.join(output_path,"models")
    if not os.path.isdir(model_file_path):
        os.makedirs(model_file_path)
    
    model_type = args.model

    
    file_path = f'Datasets/{args.dataset_name}_dataLabels.pkl'

    # Load the data and labels from the pkl file
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file)

    # Access the data and labels from the loaded dictionary
    data = data_dict['data']
    labels_array = data_dict['labels']
    
    file_path = f'Datasets/{args.dataset_name}_specs.pkl'

    # Load the data and labels from the pkl file
    with open(file_path, 'rb') as file:
        dataSpecs = pickle.load(file)
        
    SEG_SIZE = dataSpecs['SEG_SIZE']
    CHANNEL_NB = dataSpecs['CHANNEL_NB']
    CLASS_NB = dataSpecs['CLASS_NB']
    n_window, n_channel, n_data = data.shape
    
    unique_numbers_set = set(labels_array)
    num_activities = len(unique_numbers_set)
    if args.dataset_name == "PAMAP2":
        input_sequence_length = 192 #98,97
        
        Aug_index = labels_array[(labels_array == 3)] #Augmenting R windows 
        aug_data = data[(labels_array == 3),:]
        labels_array = np.append(labels_array,Aug_index)
        data = np.append(data,aug_data,0)
        
        m,n = data.shape[::2]
        n_window, n_channel, n_data = data.shape
        
    elif args.dataset_name == "SelfRegulationSCP1":
        input_sequence_length = 112
        
    elif args.dataset_name == "Shoaib":
        if model_type=="classic":
            input_sequence_length = 75
        elif model_type=="extended":
            input_sequence_length = 72 #95, 95

        
    elif args.dataset_name == "ERing":
        input_sequence_length = 8 #90,90
        
    elif args.dataset_name == "Epilepsy": #w = 275, channels = 3, n_data = 206, classes = 4, classifier tr 90.91,te 94.55, clust:2,3,4
        if model_type=="classic":
            input_sequence_length = 25
        else:
            input_sequence_length = 24 #83,93
        
    elif args.dataset_name == "WESADchest": #w= 7421, channels = 5, n_data = 200, classes = 3 ,classifier tr 94,te 98, clust:4,6,8,10
        if model_type=="classic":
            input_sequence_length = 25
        elif model_type=="extended":
            input_sequence_length = 24 #97,98
    
    elif args.dataset_name == "EMGPhysical": #w= 782, channels = 8, n_data = 200, classes = 4 ,classifier tr 93,te 77
        if model_type=="classic":
            input_sequence_length = 25
        elif model_type=="extended":
            input_sequence_length = 24 #96,89
    
    save_on = args.train
    lst = list(range(0, n_window))
    X_train_ind, X_test_ind, l, ll = train_test_split(lst, labels_array, test_size = 0.40, random_state = 0)
    X_train = data[X_train_ind,:,:]
    X_test =  data[X_test_ind,:,:]
    y_train = labels_array[X_train_ind]
    y_test = labels_array[X_test_ind]
    num_epochs = 14 #ering15, emg,epilpsey 14
    
    all_data = data.transpose(0, 2, 1).reshape(n_window, n_data, n_channel)
    n_train_data = len(X_train_ind)
    train_data = X_train.transpose(0, 2, 1).reshape(n_train_data, n_data, n_channel)
    
    train_features, test_features = prepare_data_mlp(data, labels_array, 0.4, 32, X_train_ind, X_test_ind )
    
    file_name_model = f'{args.dataset_name}_1DCNN_{args.model}.ckpt'
    model_save_path = os.path.join(model_file_path, file_name_model)
    
    if (save_on == 1):
        if model_type=="classic":
        
            cnn_model = CNN_train(train_features, test_features, device, num_epochs, CHANNEL_NB, num_activities, input_sequence_length, model_type)
            torch.save(cnn_model.state_dict(),model_save_path)
        
        elif model_type=="extended":
            cnn_model = CNN_train(train_features, test_features, device, num_epochs, CHANNEL_NB, num_activities, input_sequence_length, model_type)
            torch.save(cnn_model.state_dict(),model_save_path)
            
        elif model_type=="EarlyExit":
            thresholds = [0.90, 0.90, 0.9]  # Example thresholds
            loss_weights = [1, 1, 1, 1]
            exit_placement = [1, 2, 3]
            
            is_training = True
            
            num_exits = 3
            
            
            model = CNN1D_extended_EENN(n_channel, num_activities,
                                        input_sequence_length, thresholds,
                                        num_exits, is_training, n_data, exit_placement)
           # Loading the weights of the extended model to use for the early exit model
            #we set strict=False to load only the matching layers since EE model has extra parameters
            extended_model = f'{args.dataset_name}_1DCNN_extended.ckpt'
            saved_extended_path = os.path.join(model_file_path, extended_model)
            pretrained_state_dict = torch.load(saved_extended_path)
            model.load_state_dict(pretrained_state_dict, strict=False)
            
            model.train_model(train_features, test_features,
                              loss_weights, num_epochs)
            
            torch.save(model.state_dict(),model_save_path)



        elif model_type=="sensorAware":
            
            thresholds = [0.90, 0.90, 0.9]  # Example thresholds
            loss_weights = [1, 1, 1, 1]
            exit_placement = [1, 2, 3]
            new_data_perc = [10, 20, 30]
            
            is_training = True
            
            num_exits = 3
            
            
            model = CNN1D_extended_EENN_partialsampling(n_channel, num_activities,
                                        input_sequence_length, thresholds,
                                        num_exits, is_training, n_data, exit_placement, new_data_perc)
            
            model.train_model(train_features, test_features,
                              loss_weights, num_epochs)
            
            torch.save(model.state_dict(),model_save_path)


            
            

        
    else:
        if model_type=="classic":

            cnn_model = CNN1D(CHANNEL_NB, num_activities, input_sequence_length)
            cnn_model.load_state_dict(torch.load(model_save_path))
            cnn_model.eval()
            print("loaded model")
        
        elif model_type=="extended":
            cnn_model = CNN1D_extended(CHANNEL_NB, num_activities, input_sequence_length)
            cnn_model.load_state_dict(torch.load(model_save_path))
            cnn_model.eval()
            pytorch_total_params = sum(p.numel() for p in cnn_model.parameters())
            #print("loaded model")
            proportions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            for k in range(0,len(proportions)):
                zero_filled_data = np.zeros(data.shape)
                proportion = proportions[k]
                num_samples = int(data.shape[2] * proportion)
                zero_filled_data[:,:,:num_samples] = data[:,:,:num_samples]


                _,predicted_label_zero = torch.max(cnn_model(torch.Tensor(zero_filled_data)).data,1)
                predicted_label_zero = predicted_label_zero.detach().numpy()
                acc_zero = get_percentage_accuracy(labels_array, predicted_label_zero)
                print("Percentage",proportion*100,"Org_Acc:", acc_zero)
                
                # Calculate confusion matrix
                cm = confusion_matrix(labels_array, predicted_label_zero)
            
                # Calculate accuracy for each class
                class_accuracy = cm.diagonal() / cm.sum(axis=1)
            
                for i in range(len(class_accuracy)):
                    print(f"Class {i} Accuracy ({proportion * 100}% data): {class_accuracy[i] * 100:.2f}%")
                ###############################
                last_value_padded_data = np.repeat(data[:, :, num_samples - 1][:, :, np.newaxis], data.shape[2] - num_samples, axis=2)
                padded_data = np.concatenate((data[:, :, :num_samples], last_value_padded_data), axis=2)
            
                _, predicted_label_padded = torch.max(cnn_model(torch.Tensor(padded_data)).data, 1)
                predicted_label_padded = predicted_label_padded.detach().numpy()
            
                # Calculate confusion matrix
                cm = confusion_matrix(labels_array, predicted_label_padded)
            
                # Calculate accuracy for each class
                class_accuracy = cm.diagonal() / cm.sum(axis=1)
            
                for i in range(len(class_accuracy)):
                    print(f"Class {i} Accuracy ({proportion * 100}% data): {class_accuracy[i] * 100:.2f}%")

            zzz = 0
            
            
        elif model_type=="EarlyExit":
            #we will load the early exit network here
            z=0

        elif model_type=="sensorAware":
            #we will load the early exit + partial sampling  here

            z = 0




if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  ##General param
  parser.add_argument('--dataset_name', type=str, help="Dataset name", required=False, default="Epilepsy")
  
  parser.add_argument('--train', type=int, default=0)
  parser.add_argument('--model', type=str, help="Model type", required=False, default="extended", choices=['classic', 'extended', 'EarlyExit','sensorAware'])
  parser.add_argument('--nexits', type=int, help="Number of exits", required=False, default=2)

  args = parser.parse_args()
  main(args)