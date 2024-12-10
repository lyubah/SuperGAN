import tensorflow as tffrom tensorflow.keras import layers, modelsimport numpy as npimport picklefrom sklearn.model_selection import train_test_splitfrom sklearn.utils import class_weightfrom tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateauimport matplotlib.pyplot as pltimport osdef load_data(data_labels_path, data_specs_path):    """    Load and preprocess the dataset to match the PyTorch model's expectations.    Args:        data_labels_path (str): Path to the data labels pickle file.        data_specs_path (str): Path to the data specifications pickle file.    Returns:        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:            (X_train, X_test, y_train, y_test, CHANNEL_NB, num_activities, input_sequence_length)    """    # Load data labels    with open(data_labels_path, 'rb') as file:        data_dict = pickle.load(file)    data = data_dict['data']  # Shape: (n_window, n_channel, n_data) e.g., (275, 3, 206)    labels_array = np.array(data_dict['labels'])  # Shape: (n_window,)    # Load data specifications    with open(data_specs_path, 'rb') as file:        dataSpecs = pickle.load(file)    SEG_SIZE = dataSpecs['SEG_SIZE']          # 206    CHANNEL_NB = dataSpecs['CHANNEL_NB']      # 3    CLASS_NB = dataSpecs['CLASS_NB']          # 275    n_window, n_channel, n_data = data.shape # (275, 3, 206)    # Define number of activities and sequence length    unique_numbers_set = set(labels_array)    num_activities = len(unique_numbers_set)  # 4    input_sequence_length = n_data            # Set to 206 to match data    # Split the dataset into train and test indices with stratification    lst = list(range(0, n_window))    X_train_ind, X_test_ind, y_train, y_test = train_test_split(        lst, labels_array, test_size=0.40, random_state=0, stratify=labels_array    )    # Reshape the data for Keras model input    # Keras expects (batch_size, sequence_length, channels)    X_train = data[X_train_ind, :, :].transpose(0, 2, 1)  # (num_train, 206, 3)    X_test = data[X_test_ind, :, :].transpose(0, 2, 1)    # (num_test, 206, 3)    # Normalize the data (min-max normalization per feature)    epsilon = 1e-8  # To prevent division by zero    # Normalize X_train    X_train_min = X_train.min(axis=1, keepdims=True)  # (num_train, 1, 3)    X_train_max = X_train.max(axis=1, keepdims=True)  # (num_train, 1, 3)    X_train = (X_train - X_train_min) / (X_train_max - X_train_min + epsilon)    # Normalize X_test    X_test_min = X_test.min(axis=1, keepdims=True)    # (num_test, 1, 3)    X_test_max = X_test.max(axis=1, keepdims=True)    # (num_test, 1, 3)    X_test = (X_test - X_test_min) / (X_test_max - X_test_min + epsilon)    return X_train, X_test, y_train, y_test, CHANNEL_NB, num_activities, input_sequence_lengthdef build_CNN1D_extended_keras(input_sequence_length, CHANNEL_NB, num_activities):    """    Builds a 1D CNN model similar to the PyTorch CNN1D_extended model.    Args:        input_sequence_length (int): Length of the input sequence (e.g., 206).        CHANNEL_NB (int): Number of input channels (e.g., 3).        num_activities (int): Number of output classes (e.g., 4).    Returns:        tf.keras.Model: Compiled Keras model.    """    # Define the input with shape (sequence_length, channels)    input_shape = (input_sequence_length, CHANNEL_NB)    inputs = tf.keras.Input(shape=input_shape, name='Input_Layer')    # First Conv1D block    x = layers.Conv1D(filters=8, kernel_size=3, padding='same', name='Conv1D_1')(inputs)    x = layers.ReLU(name='ReLU_1')(x)    x = layers.MaxPooling1D(pool_size=2, name='MaxPool1D_1')(x)    # Second Conv1D block    x = layers.Conv1D(filters=16, kernel_size=3, padding='same', name='Conv1D_2')(x)    x = layers.ReLU(name='ReLU_2')(x)    x = layers.MaxPooling1D(pool_size=2, name='MaxPool1D_2')(x)    # Third Conv1D block    x = layers.Conv1D(filters=32, kernel_size=3, padding='same', name='Conv1D_3')(x)    x = layers.ReLU(name='ReLU_3')(x)    x = layers.MaxPooling1D(pool_size=2, name='MaxPool1D_3')(x)    # Fourth Conv1D block    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', name='Conv1D_4')(x)    x = layers.ReLU(name='ReLU_4')(x)    x = layers.MaxPooling1D(pool_size=2, name='MaxPool1D_4')(x)    # Fifth Conv1D block    x = layers.Conv1D(filters=128, kernel_size=3, padding='same', name='Conv1D_5')(x)    x = layers.ReLU(name='ReLU_5')(x)    x = layers.MaxPooling1D(pool_size=2, name='MaxPool1D_5')(x)    # Flatten the output    x = layers.Flatten(name='Flatten')(x)    # Fully Connected Layers    x = layers.Dense(64, activation='relu', name='Dense_1')(x)    outputs = layers.Dense(num_activities, activation='softmax', name='Output_Layer')(x)    # Create the model    model = models.Model(inputs=inputs, outputs=outputs, name='CNN1D_Extended_Keras')    return modelif __name__ == "__main__":    # Define file paths    data_labels_path = "/Users/Lerberber/Desktop/Bhat/SuperGans/CNN1_scripts/Datasets/Epilepsy_dataLabels.pkl"    data_specs_path = "/Users/Lerberber/Desktop/Bhat/SuperGans/CNN1_scripts/Datasets/Epilepsy_specs.pkl"    save_dir = "/Users/Lerberber/Desktop/Bhat/SuperGans/CNN1_scripts/Models"    # Create the save directory if it doesn't exist    if not os.path.exists(save_dir):        os.makedirs(save_dir)        print(f"Created directory: {save_dir}")    else:        print(f"Save directory already exists: {save_dir}")    # Load data    X_train, X_test, y_train, y_test, CHANNEL_NB, num_activities, input_sequence_length = load_data(        data_labels_path, data_specs_path    )    # Display loaded data shapes    print(f"X_train shape: {X_train.shape}")  # Expected: (165, 206, 3)    print(f"X_test shape: {X_test.shape}")    # Expected: (110, 206, 3)    print(f"y_train shape: {y_train.shape}")  # Expected: (165,)    print(f"y_test shape: {y_test.shape}")    # Expected: (110,)    print(f"CHANNEL_NB: {CHANNEL_NB}")        # Expected: 3    print(f"num_activities: {num_activities}")# Expected: 4    print(f"input_sequence_length: {input_sequence_length}")  # Expected: 206    # Compute class weights to handle class imbalance    class_weights_vals = class_weight.compute_class_weight('balanced',                                                           classes=np.unique(y_train),                                                           y=y_train)    class_weights_dict = {i : class_weights_vals[i] for i in range(num_activities)}    print(f"Class Weights: {class_weights_dict}")    # Build the model    model = build_CNN1D_extended_keras(input_sequence_length, CHANNEL_NB, num_activities)    # Compile the model    model.compile(optimizer="adam",                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),                  metrics=["accuracy"])    # Display the model summary    model.summary()    # Define callbacks    early_stopping = EarlyStopping(monitor='val_accuracy',                                    patience=10,                                    restore_best_weights=True,                                   verbose=1)    # Define the full paths for saving models    best_model_filename = "best_CNN1D_Extended_Keras_model.h5"    final_model_filename = "CNN1D_Extended_Keras_model_final.h5"    best_model_path = os.path.join(save_dir, best_model_filename)    final_model_path = os.path.join(save_dir, final_model_filename)    print(f"Best model will be saved to: {best_model_path}")    print(f"Final model will be saved to: {final_model_path}")    # Initialize ModelCheckpoint with the absolute path    model_checkpoint = ModelCheckpoint(best_model_path,                                       monitor='val_accuracy',                                       save_best_only=True,                                       verbose=1)    # Initialize ReduceLROnPlateau with the absolute path    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',                                                      factor=0.5,                                                      patience=5,                                                      min_lr=1e-6,                                                      verbose=1)    # Combine all callbacks    callbacks = [early_stopping, model_checkpoint, reduce_lr]    # Train the model initially without adversarial training    history = model.fit(X_train, y_train,                         validation_data=(X_test, y_test),                         epochs=50,  # Start with 50 epochs                        batch_size=32,                        callbacks=callbacks,                        class_weight=class_weights_dict)    # Evaluate the initial model    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)    print(f"Initial Test Loss: {loss:.4f}")    print(f"Initial Test Accuracy: {accuracy:.4f}")    # Save the final model to the specified directory    try:        model.save(final_model_path)        print(f"Model training complete and saved successfully to {final_model_path}.")    except Exception as e:        print(f"Error saving the final model: {e}")    # Verify that the models have been saved    print("\nVerifying saved model files:")    if os.path.exists(best_model_path):        print(f"Best model exists: {best_model_path}")    else:        print(f"Best model NOT found: {best_model_path}")    if os.path.exists(final_model_path):        print(f"Final model exists: {final_model_path}")    else:        print(f"Final model NOT found: {final_model_path}")    # List all .h5 files in the save directory    print("\nListing all .h5 files in the Models directory:")    h5_files = [file for file in os.listdir(save_dir) if file.endswith('.h5')]    if h5_files:        for file in h5_files:            file_path = os.path.join(save_dir, file)            print(f"- {file_path} (Size: {os.path.getsize(file_path)} bytes)")    else:        print("No .h5 files found in the Models directory.")    # Perform a forward pass with dummy data to verify    X_dummy = np.random.rand(1, input_sequence_length, CHANNEL_NB).astype('float32')  # (1, 206, 3)    y_dummy = np.array([0])  # Example target    prediction = model.predict(X_dummy)    print(f"\nPrediction shape: {prediction.shape}")  # Should be (1, 4)    print(f"Prediction: {prediction}")             # Softmax probabilities    # Plot training & validation accuracy values    plt.figure(figsize=(12, 4))    plt.subplot(1, 2, 1)    plt.plot(history.history['accuracy'], label='Train Accuracy')    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')    plt.title('Model Accuracy Over Epochs')    plt.xlabel('Epoch')    plt.ylabel('Accuracy')    plt.legend()    # Plot training & validation loss values    plt.subplot(1, 2, 2)    plt.plot(history.history['loss'], label='Train Loss')    plt.plot(history.history['val_loss'], label='Validation Loss')    plt.title('Model Loss Over Epochs')    plt.xlabel('Epoch')    plt.ylabel('Loss')    plt.legend()    plt.tight_layout()    plt.show()