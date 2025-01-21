import numpy as np

def key_to_one_hot(y, values):
	one_hot = np.zeros((y.size, values.size), dtype=np.float64)
	one_hot[np.arange(y.size), y] = 1

	return one_hot

def cross_entropy_to_key(y):
	return np.argmax(y, axis = 1)

def get_accuracy_cross_entropy(predictions, y):
	return get_accuracy_key(cross_entropy_to_key(predictions), cross_entropy_to_key(y))

def get_accuracy_key(predictions, y):
	# Assumes index labels, not one-hot
    return np.sum(predictions == y) / y.size