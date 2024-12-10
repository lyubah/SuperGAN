import h5pyimport numpy as npdef inspect_h5(file_path):    with h5py.File(file_path, 'r') as f:        print(f"Inspecting file: {file_path}")        for key in f.keys():            print(f"Dataset: {key}")            print(f"Shape: {f[key].shape}")            print(f"Dtype: {f[key].dtype}")            print(f"First few values of {key}: {f[key][0]}\n")        return {key: f[key][:] for key in f.keys()}# File pathsoutput_h5_path = '/Users/Lerberber/Desktop/Bhat/SuperGans/data/Epilepsy_preprocessed.h5'epilepsy_file = "/Users/Lerberber/Desktop/Bhat/SuperGans/data/Epilepsy_preprocessed.h5"accelerometer_file = "/Users/Lerberber/Desktop/Bhat/SuperGans/sports_data_accelerometer.h5"  gyroscope_file = "/Users/Lerberber/Desktop/Bhat/SuperGans/sports_data_gyroscope.h5"# Inspect datasetsepilepsy_data = inspect_h5(epilepsy_file)accelerometer_data = inspect_h5(accelerometer_file)gyro_file = inspect_h5(gyroscope_file)# Compare structuresprint("\n--- Comparison ---")for key in accelerometer_data.keys():    if key in epilepsy_data:        print(f"Dataset: {key}")        print(f"Accelerometer shape: {accelerometer_data[key].shape}, Epilepsy shape: {epilepsy_data[key].shape}")        print(f"Shape matches: {accelerometer_data[key].shape == epilepsy_data[key].shape}")        print(f"Sample values comparison:")        print(f"Accelerometer first entry: {accelerometer_data[key][0]}")        print(f"Epilepsy first entry: {epilepsy_data[key][0]}\n")    else:        print(f"{key} missing in epilepsy data.")