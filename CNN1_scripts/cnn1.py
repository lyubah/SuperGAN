# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# verify_cnn1_extended.py

# This script verifies the performance of the pre-trained CNN1 Extended model
# on the Epilepsy dataset. It loads the data, preprocesses it, loads the model,
# evaluates it on the test set, and reports performance metrics.

# Author: Dina Hussein
# Created on: April 27, 2024
# """

# import os
# import pickle
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.metrics import classification_report, accuracy_score
# import numpy as np

# from torch.utils.data import DataLoader
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.data import DataLoader, TensorDataset
# import torch.utils.data as dl


# # Set random seeds for reproducibility
# torch.manual_seed(123)
# np.random.seed(123)

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')



# class mlp_dataset(dl.Dataset):
#     def __init__(self, X, y):
#         self.X = X # input accelerometer data
#         self.X = self.X.astype('float32')
#         self.y = y # output stretch data that we need
#         self.y = self.y.astype(int)

    
#     def __len__(self):
#         return np.shape(self.X)[0]

#     def __getitem__(self, idx):
#         return [self.X[idx], self.y[idx]]

#     # get indexes for train and test rows
#     def get_splits(self, test_split):
#         # determine sizes
#         n_data = np.shape(self.X)[0]
#         test_size = round(test_split * n_data)
#         train_size = len(self.X) - test_size
#         # calculate the split
#         return dl.random_split(self, [train_size, test_size])
    
    

# def prepare_data_mlp(feature_matrix, label_vector, test_split, bs,train_indices,test_indices):
#     dataset = mlp_dataset(feature_matrix, label_vector)
#     train, test = dataset.get_splits(test_split)
#     train.indices = train_indices
#     test.indices = test_indices

#     train_dl = dl.DataLoader(train, batch_size = bs, shuffle=True)
#     test_dl = dl.DataLoader(test, batch_size = bs, shuffle=False)
#     # train_indices = train
#     # test_indices = test

#     return train_dl, test_dl 


# # -----------------------------------
# # Data Loading and Preprocessing
# # -----------------------------------


# def load_data(data_labels_path, data_specs_path):
#     """
#     Load and preprocess the dataset from pickle files.

#     Args:
#         data_labels_path (str): Path to the data labels pickle file.
#         data_specs_path (str): Path to the data specifications pickle file.

#     Returns:
#         Tuple[DataLoader, DataLoader, int, int, int, list]:
#             (train_loader, test_loader, CHANNEL_NB, num_activities, input_sequence_length, class_names)
#     """
#     # Load data labels
#     with open(data_labels_path, 'rb') as file:
#         data_dict = pickle.load(file)
#     data = data_dict['data']  # Shape: (n_window, n_channel, n_data)
#     labels_array = np.array(data_dict['labels'])  # Shape: (n_window,)

#     # Load data specifications
#     with open(data_specs_path, 'rb') as file:
#         dataSpecs = pickle.load(file)

#     SEG_SIZE = dataSpecs['SEG_SIZE']
#     CHANNEL_NB = dataSpecs['CHANNEL_NB']
#     CLASS_NB = dataSpecs['CLASS_NB']
#     n_window, n_channel, n_data = data.shape
#     unique_classes = sorted(list(set(labels_array)))
#     num_activities = len(unique_classes)

#     print(f'Dataset Loaded: {n_window} samples, {n_channel} channels, {n_data} data points per channel')
#     print(f'Number of Classes: {num_activities}')
#     print(f'Classes: {unique_classes}')

#     # Define input sequence length based on model type
#     input_sequence_length = dataSpecs.get('INPUT_SEQUENCE_LENGTH', 24)

#     # Perform 60:40 train-test split with stratification
#     X = data  # Shape: (n_window, n_channel, n_data)
#     y = labels_array  # Shape: (n_window,)

#     # Generate train-test indices
#     lst = list(range(0, n_window))
#     X_train_ind, X_test_ind, y_train, y_test = train_test_split(
#         lst, y, test_size=0.40, random_state=0, stratify=y
#     )
#     train_features, test_features = prepare_data_mlp(
#         data, labels_array, test_split=0.4, bs=32, 
#         train_indices=X_train_ind, test_indices=X_test_ind
#     )

#     return train_features, test_features, CHANNEL_NB, num_activities, input_sequence_length, unique_classes

# # -----------------------------------
# # Model Definition
# # -----------------------------------

# class CNN1D_extended(nn.Module):
#     def __init__(self, in_channels, out_classes, input_sequence_length):
#         super(CNN1D_extended, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv1d(64, 128, kernel_size=3, padding=1)


#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool1d(kernel_size=2)
#         self.fc1_input_size = 32 * input_sequence_length  #75 for shoaib, 192 for pamap
        
#         self.fc1 = nn.Linear(self.fc1_input_size, 64)
#         self.fc2 = nn.Linear(64, out_classes)

#     def forward(self, x):
#         # x has shape (batch_size, in_channels, sequence_length)
#         print(x.shape)
#         x = self.conv1(x)
#         print(x.shape)
#         x = self.relu(x)
#         print(x.shape)
#         x = self.maxpool(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.conv4(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.conv5(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x
    
# # -----------------------------------
# # Model Loading and Evaluation
# # -----------------------------------

# def load_pretrained_model(model_path, in_channels, out_classes, input_sequence_length):
#     model = CNN1D_extended(in_channels, out_classes, input_sequence_length)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
#     print(f'Pre-trained model loaded from {model_path}')
#     return model

# # def evaluate_model(model, test_loader, device, class_names):
# #     all_preds = []
# #     all_labels = []
# #     with torch.no_grad():
# #         for batch_data, batch_labels in test_loader:
# #             batch_data = batch_data.to(device)
# #             outputs = model(batch_data)
# #             _, predicted = torch.max(outputs.data, 1)
# #             all_preds.extend(predicted.cpu().numpy())
# #             all_labels.extend(batch_labels.cpu().numpy())
# #     print(f'Accuracy: {accuracy_score(all_labels, all_preds):.2f}')
# #     print(classification_report(all_labels, all_preds, target_names=class_names))
# #     return accuracy_score(all_labels, all_preds)


# def evaluate_model(model, test_loader, device, class_names=None):
#     """
#     Evaluate the model on the test dataset and print accuracy and classification report.

#     Args:
#         model (nn.Module): Trained model.
#         test_loader (DataLoader): Test DataLoader.
#         device (torch.device): Device to run the evaluation on.
#         class_names (list or None): List of class names.

#     Returns:
#         float: Accuracy score on the test dataset.
#     """
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch_data, batch_labels in test_loader:
#             batch_data = batch_data.to(device)
#             outputs = model(batch_data)
#             _, predicted = torch.max(outputs.data, 1)
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(batch_labels.cpu().numpy())

#     # Ensure class_names is valid and a list of strings
#     if class_names is None or not all(isinstance(name, str) for name in class_names):
#         unique_labels = np.unique(all_labels)  # Find unique labels
#         class_names = [f"Class {int(label)}" for label in unique_labels]  # Convert to strings

#     print(f'Accuracy: {accuracy_score(all_labels, all_preds):.2f}')
#     print(classification_report(all_labels, all_preds, target_names=class_names))
#     return accuracy_score(all_labels, all_preds)


# # -----------------------------------
# # Main Function
# # -----------------------------------

# def main():
#     base_dir = '/Users/Lerberber/Desktop/Bhat/SuperGans/CNN1_scripts/Datasets'
#     data_labels_path = os.path.join(base_dir, 'Epilepsy_dataLabels.pkl')
#     data_specs_path = os.path.join(base_dir, 'Epilepsy_specs.pkl')
#     model_path = os.path.join(base_dir, 'Epilepsy_1DCNN_extended.ckpt')

#     train_loader, test_loader, in_channels, out_classes, input_sequence_length, class_names = load_data(
#         data_labels_path, data_specs_path
#     )

#     model = load_pretrained_model(model_path, in_channels, out_classes, input_sequence_length)

#     evaluate_model(model, test_loader, device, class_names)

# if __name__ == "__main__":
#     main()
import torch
import torch.nn as nn

class CNN1D_extended(nn.Module):
    def __init__(self, in_channels, out_classes, input_sequence_length):
        super(CNN1D_extended, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Adjusted fc1_input_size based on input_sequence_length=24 and number of poolings=4
        # After 4 poolings: 24 -> 12 -> 6 -> 3 -> 1
        # Thus, fc1_input_size = 128 * 1 = 128
        self.fc1_input_size = 128 * 1  # 128 filters * final sequence length

        self.fc1 = nn.Linear(self.fc1_input_size, 64)
        self.fc2 = nn.Linear(64, out_classes)

    def forward(self, x):
        # x has shape (batch_size, in_channels, sequence_length)
        print(f"Input shape: {x.shape}")

        # Conv1
        x = self.conv1(x)
        print(f"After Conv1: {x.shape}")
        x = self.relu(x)
        print(f"After ReLU1: {x.shape}")
        x = self.maxpool(x)
        print(f"After MaxPool1: {x.shape}")

        # Conv2
        x = self.conv2(x)
        print(f"After Conv2: {x.shape}")
        x = self.relu(x)
        print(f"After ReLU2: {x.shape}")
        x = self.maxpool(x)
        print(f"After MaxPool2: {x.shape}")

        # Conv3
        x = self.conv3(x)
        print(f"After Conv3: {x.shape}")
        x = self.relu(x)
        print(f"After ReLU3: {x.shape}")
        x = self.maxpool(x)
        print(f"After MaxPool3: {x.shape}")

        # Conv4
        x = self.conv4(x)
        print(f"After Conv4: {x.shape}")
        x = self.relu(x)
        print(f"After ReLU4: {x.shape}")
        x = self.maxpool(x)
        print(f"After MaxPool4: {x.shape}")

        # Conv5
        x = self.conv5(x)
        print(f"After Conv5: {x.shape}")
        x = self.relu(x)
        print(f"After ReLU5: {x.shape}")
        # Removed the 5th pooling layer to prevent sequence_length=0
        # x = self.maxpool(x)
        # print(f"After MaxPool5: {x.shape}")

        # Flatten
        x = x.view(x.size(0), -1)
        print(f"After Flatten: {x.shape}")

        # Fully Connected Layers
        x = self.fc1(x)
        print(f"After FC1: {x.shape}")
        x = self.relu(x)
        print(f"After ReLU FC1: {x.shape}")
        x = self.fc2(x)
        print(f"After FC2: {x.shape}")

        return x

# Example usage
if __name__ == "__main__":
    # Define model parameters
    in_channels = 3
    out_classes = 4
    input_sequence_length = 24

    # Instantiate the model
    model = CNN1D_extended(in_channels, out_classes, input_sequence_length)

    # Create dummy input (batch_size=1, in_channels=3, sequence_length=24)
    dummy_input = torch.randn(1, in_channels, input_sequence_length)

    # Perform a forward pass
    output = model(dummy_input)

    # Print final output
    print(f"Final Output: {output.shape}")

