"""
Generates a trained LSTM classifier for a dataset to use with SuperGAN
Usage: python3 train_simple_lstm.py dataset_name.h5 classifier_name.h5
"""

import sys
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split


def create_classifier_model(number_of_classes: int):
    """
    Creates the simplest possible LSTM Model for a classifier.

    :param number_of_classes: The number of classes to add
    :return:
    """
    lstm_model = Sequential(name="classifier")
    lstm_model.add(LSTM(100, activation='tanh'))
    lstm_model.add(Dense(number_of_classes, activation='softmax'))
    lstm_model.compile(optimizer='rmsprop', loss='mse')
    return lstm_model


# Evaluate accuracy for multiclass classification
def evaluate_model(predictions, actual):
    n = actual.shape[0]
    tc, fc = (0.0, 0.0)
    for i in range(0, n):
        p = predictions[i]
        a = np.argmax(actual[i])

        # Did we classify correctly?
        if p == a:
            tc += 1.0
        else:
            fc += 1.0

    print("Total Accuracy: ", tc / (tc + fc))


if __name__ == '__main__':
    # Sanity check:
    if len(sys.argv) < 3:
        print("Usage: python3 train_simple_lstm.py dataset_name.h5 classifier_name.h5")
        exit(1)

    # Load dataset
    dataset = h5py.File(sys.argv[1], 'r')
    x = dataset['X']
    y = dataset['y_onehot']

    # Split dataset into 30% testing, 70% training
    x_train, x_test, y_train, y_test = train_test_split(
        np.asarray(x),
        np.asarray(y),
        test_size=0.3
    )

    # Train model on training data, validate on testing data
    num_classes = y.shape[1]
    model = create_model(num_classes)
    fitted = model.fit(
        x_train, y_train,
        epochs=300, batch_size=100,
        validation_data=(x_test, y_test),
        verbose=2, shuffle=False
    )

    # Evaluate, so we know the model is decent
    pred = np.argmax(model.predict(x_test), axis=-1)

    print("Model Performance:")
    print("------------------")
    evaluate_model(pred, y_test)

    # Save model.
    # Note: we could also save x_test and y_test as a holdout set for TSTR
    model.save(sys.argv[2])
