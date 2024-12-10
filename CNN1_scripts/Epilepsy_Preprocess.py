import os
import pickle
import numpy as np
import h5py

def preprocess_data_for_supergan(data_labels_path, data_specs_path, output_h5_path):
    """
    Preprocess data for SuperGAN framework and save it in HDF5 format.

    Args:
        data_labels_path (str): Path to data labels pickle file.
        data_specs_path (str): Path to data specifications pickle file.
        output_h5_path (str): Path to save preprocessed HDF5 file.

    Returns:
        dict: Metadata about the preprocessed data.
    """
    # Load data labels
    with open(data_labels_path, 'rb') as file:
        data_dict = pickle.load(file)
    data = data_dict['data']  # Shape: (n_window, n_channel, n_data)
    labels_array = np.array(data_dict['labels'])  # Shape: (n_window,)

    # Load data specifications
    with open(data_specs_path, 'rb') as file:
        dataSpecs = pickle.load(file)

    # Extract specifications
    SEG_SIZE = 24
    CHANNEL_NB = dataSpecs['CHANNEL_NB']
    # Dynamically determine CLASS_NB based on the actual unique labels in the dataset
    unique_numbers_set = set(labels_array)
    CLASS_NB = len(unique_numbers_set)
    n_window, n_channel, n_data = data.shape

    # Define input sequence length
    input_sequence_length = 24

    print(f"Dataset Loaded: {n_window} samples, {n_channel} channels, {n_data} data points per channel")
    print(f"Number of Classes: {CLASS_NB}")

    # Reshape the data for SuperGAN: (num_samples, seg_length, num_channels)
    all_data = data.transpose(0, 2, 1).reshape(n_window, n_data, n_channel)

    # Prepare integer labels and one-hot encoded labels
    y_gan = labels_array
    y_gan_onehot = np.zeros((n_window, CLASS_NB), dtype=np.float32)
    for i, lbl in enumerate(y_gan):
        y_gan_onehot[i, lbl] = 1.0

    # Save the data into HDF5 format
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('X', data=all_data)  # Feature data
        f.create_dataset('y', data=y_gan)  # Integer labels
        f.create_dataset('y_onehot', data=y_gan_onehot)  # One-hot encoded labels
        f.attrs['input_sequence_length'] = input_sequence_length
        f.attrs['SEG_SIZE'] = SEG_SIZE
        f.attrs['CHANNEL_NB'] = CHANNEL_NB
        f.attrs['CLASS_NB'] = CLASS_NB

    print(f"Preprocessed data saved to {output_h5_path}")

    return {
        "X_shape": all_data.shape,
        "y_shape": y_gan.shape,
        "y_onehot_shape": y_gan_onehot.shape,
        "input_sequence_length": input_sequence_length,
        "SEG_SIZE": SEG_SIZE,
        "CHANNEL_NB": CHANNEL_NB,
        "CLASS_NB": CLASS_NB
    }


# Example usage
def main():
    base_dir = '/Users/Lerberber/Desktop/Bhat/SuperGans/CNN1_scripts/Datasets'
    data_labels_path = os.path.join(base_dir, 'Epilepsy_dataLabels.pkl')
    data_specs_path = os.path.join(base_dir, 'Epilepsy_specs.pkl')
    output_h5_path = '/Users/Lerberber/Desktop/Bhat/SuperGans/data/Epilepsy_preprocessed.h5'

    # Preprocess data for SuperGAN
    metadata = preprocess_data_for_supergan(data_labels_path, data_specs_path, output_h5_path)

    # Summary of processed data
    print("\n--- Summary of Preprocessed Data ---")
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
