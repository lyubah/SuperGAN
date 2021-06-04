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


# Simplest possible LSTM model. 
def create_model(num_classes):
    model = Sequential(name = "classifier")
    model.add(LSTM(128, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='mae')
    return model


# Evaluate a model according to both binary metrics, where none is the negative
# class and everything else is the positive class, and a multiclass metric,
# where accuracy between attack classes is considered.
def evaluate_model(predictions, actual):
	n = actual.shape[0]
	tp,tn,tc,fp,fn,fc=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	for i in range(0, n):
		p = predictions[i]
		a = np.argmax(actual[i])
		# Did we classify correctly?
		if p == a:
			tc += 1.0
		else:
			fc += 1.0
		# Did we at least classify binary correctly?
		p_pos = (p != 0)
		a_pos = (a != 0)
		if p_pos:
			if a_pos:
				tp += 1.0
			else:
				fp += 1.0
		else:
			if a_pos:
				fn += 1.0
			else:
				tn += 1.0
	print("Total Accuracy: ", tc/(tc+fc))
	print("Binary Accuracy: ", (tp+tn)/(tp+tn+fn+fp))
	print("Precision: ", tp/(tp+fp))
	print("Recall: ", tp/(tp+fn))
	print("FPR: ", fp/(fp+tn))


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
	    test_size = 0.3
    )

    # Train model on training data, validate on testing data
    num_classes = y.shape[1]
    model = create_model(num_classes)
    fitted = model.fit(
	    x_train, y_train,
	    epochs=30, batch_size=200,
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
