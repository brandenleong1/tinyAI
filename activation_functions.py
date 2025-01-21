import numpy as np

def sigmoid(x, derivative = False):
	if not derivative:
		return 1.0 / (1.0 + np.exp(-x))
	else:
		return sigmoid(x) * (1.0 - sigmoid(x))

def hard_sigmoid(x, derivative = False):
	if not derivative:
		return np.piecewise(x, [x < -3, x >= -3 and x <= 3, x > 3], [0.0, lambda x: x / 0.6 + 0.5, 1.0])
	else:
		return np.piecewise(x, [x < -3, x >= -3 and x <= 3, x > 3], [0.0, lambda x: 1.0 / 0.6, 0.0])

def silu(x, derivative = False):
	if not derivative:
		return x * sigmoid(x)
	else:
		return silu(x) + sigmoid(x)(1.0 - silu(x))

def hard_silu(x, derivative = False):
	if not derivative:
		return np.piecewise(x, [x < -3, x >= -3 and x <= 3, x > 3], [0.0, lambda x: x * (x + 3.0) / 6.0, lambda x: x])
	else:
		return np.piecewise(x, [x < -3, x >= -3 and x <= 3, x > 3], [0.0, lambda x: (x * 2.0 + 3.0) / 6.0, 1.0])

def tanh(x, derivative = False):
	if not derivative:
		return np.tanh(x)
	else:
		return 1.0 - np.tanh(x) ** 2

def relu(x, derivative = False):
	if not derivative:
		return np.maximum(0, x)
	else:
		return np.piecewise(x, [x < 0, x >= 0], [0.0, 1.0])

def leaky_relu(x, derivative = False, negative_slope = 0.2):
	if not derivative:
		return np.maximum(negative_slope * x, x)
	else:
		return np.piecewise(x, [x < 0, x >= 0], [negative_slope, 1.0])

def relu6(x, derivative = False):
	if not derivative:
		return np.minimum(np.maximum(0, x), 6.0)
	else:
		return np.piecewise(x, [x < 0 or x > 6, x >= 0 and x <= 6], [0.0, 1.0])

def softmax(x, derivative = False):
	if not derivative:
		return np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
	else:
		return np.ones_like(x)

def log_softmax(x, derivative = False):
	if not derivative:
		return np.log(softmax(x))
	else:
		return 1.0 / softmax(x)

def softplus(x, derivative = False):
	if not derivative:
		return np.log(np.exp(x) + 1.0)
	else:
		return np.exp(x) / (np.exp(x) + 1.0)

def softsign(x, derivative = False):
	if not derivative:
		return x / (np.abs(x) + 1.0)
	else:
		return 1.0 / ((np.abs(x) + 1.0) ** 2)

def elu(x, derivative = False, alpha = 1.0):
	if not derivative:
		return np.piecewise(x, [x < 0, x >= 0], [lambda x: alpha * np.exp(x) - 1.0, lambda x: x])
	else:
		return np.piecewise(x, [x < 0, x >= 0], [lambda x: alpha * np.exp(x), 1.0])

def mish(x, derivative = False):
	if not derivative:
		return x * tanh(softplus(x))
	else:
		omega = np.exp(3.0 * x) + 4.0 * np.exp(2.0 * x) + (6.0 + 4.0 * x) * np.exp(x) + 4.0 * (1.0 + x)
		delta = 1.0 + np.power((np.exp(x) + 1.0), 2)
		derivative = np.exp(x) * omega / np.power(delta, 2)
		return derivative

def linear(x, derivative = False):
	if not derivative:
		return x
	else:
		return np.ones_like(x)

def exponential(x, derivative = False):
	if not derivative:
		return np.exp(x)
	else:
		return np.exp(x)

