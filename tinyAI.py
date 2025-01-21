import numpy as np
import progressbar as pb
import matplotlib.pyplot as plt
from typing import Callable, List, Optional

import activation_functions as af
import functions
from utils import *

class Layer(object):
	def __init__(self) -> None:
		self.info: dict
		self.activation: Callable
		self.weights: np.ndarray
		self.biases: np.ndarray
		self.raw_outputs: np.ndarray
		self.inputs: np.ndarray
		self.outputs: np.ndarray
		self.adjustment_weights: np.ndarray
		self.adjustment_biases: np.ndarray
		self.input_loss: np.ndarray
		self.velocity_weights: np.ndarray
		self.velocity_biases: np.ndarray
		self.is_output_layer: bool

		self.input_shape: tuple
		self.output_shape: tuple

	def compile(self, *args, **kwargs):
		raise NotImplementedError

	def forward(self, *args, **kwargs):
		raise NotImplementedError

	def backward(self, *args, **kwargs):
		raise NotImplementedError

	def update(self, *args, **kwargs):
		raise NotImplementedError

	def get_weights(self, *args, **kwargs):
		raise NotImplementedError

	def load_weights(self, *args, **kwargs):
		raise NotImplementedError

class Dense(Layer):
	def __init__(self, nodes: int, activation: Optional[Callable] = None) -> None:
		self.info =					{'nodes': nodes}
		self.activation =			activation or af.linear

	def compile(self, input_shape: tuple[int], is_output_layer: bool = False, *args, **kwargs) -> None:
		self.input_shape =			input_shape
		self.output_shape =			(self.info['nodes'],)

		fan_in = self.input_shape[0]
		fan_out = self.output_shape[0]

		self.weights =				(np.random.rand(input_shape[0], self.info['nodes']) - 0.5) * 2.0 * np.sqrt(6.0 / (fan_in + fan_out))
		self.biases =				np.zeros((self.info['nodes'],))
		self.raw_outputs =			np.empty(())
		self.inputs =				np.empty(())
		self.outputs =				np.empty(())
		self.adjustment_weights =	np.empty(())
		self.adjustment_biases =	np.empty(())
		self.input_loss =			np.empty(())
		self.velocity_weights =		np.zeros(self.weights.shape)
		self.velocity_biases =		np.zeros(self.biases.shape)
		self.is_output_layer =		is_output_layer

	def forward(self, x: np.ndarray, *args, **kwargs) -> None:
		self.inputs = x
		self.raw_outputs = (x @ self.weights) + self.biases # (64, 784) @ (784, 16) = (64, 16) + (1, 16)
		self.outputs = self.activation(self.raw_outputs)

	def backward(self, y: np.ndarray, next_layer: Layer, *args, **kwargs) -> None:
		if self.is_output_layer:
			output_loss = self.outputs - y
		else:
			output_loss = next_layer.input_loss

		adjustment = output_loss * self.activation(self.raw_outputs, True)

		self.adjustment_weights = self.inputs.T @ adjustment
		self.adjustment_biases = np.sum(adjustment, axis = 0).reshape(1, -1)

		self.input_loss = adjustment @ self.weights.T

	def update(self, learning_rate: float, momentum: float, *args, **kwargs) -> None:
		self.velocity_weights = momentum * self.velocity_weights - (learning_rate * self.adjustment_weights / self.inputs.shape[0])
		self.velocity_biases = momentum * self.velocity_biases - (learning_rate * self.adjustment_biases / self.inputs.shape[0])

		self.weights = self.weights + self.velocity_weights
		self.biases = self.biases + self.velocity_biases

	def get_weights(self, *args, **kwargs) -> list[np.ndarray]:
		return [self.weights, self.biases]

	def load_weights(self, weights: list[np.ndarray], *args, **kwargs):
		if len(weights) >= 1:
			if weights[0].shape == self.weights.shape:
				self.weights = weights[0]
			else:
				raise ValueError('Mismatched weights ' + str(weights[0].shape) + ' and ' + str(self.weights.shape))
		if len(weights) >= 2:
			if weights[1].shape == self.biases.shape:
				self.biases = weights[1]
			else:
				raise ValueError('Mismatched weights ' + str(weights[1].shape) + ' and ' + str(self.biases.shape))

class Conv2D(Layer):
	def __init__(self,
			filters: int,
			kernel_size: int | tuple[int, int],
			strides: int | tuple[int, int] = (1, 1),
			padding: str = 'valid',
			dilation_rate: int | tuple[int, int] = (1, 1),
			groups: int = 1,
			activation: Optional[Callable] = None
		) -> None:

		if padding.lower() not in ['valid', 'same']:
			raise ValueError(r'The argument `padding` must be one of {\'valid\', \'same\'}. Received: ' + padding)

		self.info = 			{
			'filters': filters,
			'kernel_size': kernel_size if type(kernel_size) == tuple else (kernel_size,) * 2,
			'strides': strides if type(strides) == tuple else (strides,) * 2,
			'padding': padding,
			'dilation_rate': dilation_rate if type(dilation_rate) == tuple else (dilation_rate,) * 2,
			'groups': groups
		}
		self.activation =		activation or af.linear

	def compile(self, input_shape: tuple[int, int, int], is_output_layer: bool = False, *args, **kwargs) -> None:
									# (height, width, channels)
		self.input_shape =			input_shape
									# (new_height, new_width, filters)
		self.output_shape =			functions.conv_shape(
										input_shape = input_shape[0:2],
										filter_shape = self.info['kernel_size'],
										strides = self.info['strides'],
										padding = self.info['padding'],
										dilation = self.info['dilation_rate']
									) + (self.info['filters'],)

		if input_shape[2] % self.info['groups'] != 0:
			raise ValueError(r'Index 2 of the argument `input_shape` ' + str(input_shape) + ' must be divisible by the argument `groups` (' + str(self.info['groups']) + ')')
		if self.info['filters'] % self.info['groups'] != 0:
			raise ValueError(r'The argument `filters` (' + str(self.info['filters']) + ') must be divisible by the argument `groups` (' + str(self.info['groups']) + ')')

		receptive_field_size =		np.prod(self.info['kernel_size'])
		fan_in =					(self.input_shape[2] // self.info['groups']) * receptive_field_size
		fan_out =					(self.info['filters']) * receptive_field_size

		self.weights =				(np.random.rand(*self.info['kernel_size'], self.input_shape[2] // self.info['groups'], self.info['filters']) - 0.5) * 2.0 * np.sqrt(6.0 / (fan_in + fan_out))
		self.biases =				np.zeros((self.info['filters'],))
		self.raw_outputs =			np.empty(())
		self.inputs =				np.empty(())
		self.outputs =				np.empty(())
		self.adjustment_weights =	np.empty(())
		self.adjustment_biases =	np.empty(())
		self.input_loss =			np.empty(())
		self.velocity_weights =		np.zeros(self.weights.shape)
		self.velocity_biases =		np.zeros(self.biases.shape)
		self.is_output_layer =		is_output_layer

	def forward(self, x: np.ndarray, *args, **kwargs) -> None:
		# (batch_size, height, width, channels)
		self.inputs = x

		batch_size = x.shape[0]
		channels = x.shape[3]
		filter_size = self.info['filters']
		groups = self.info['groups']

		group_channel_size = channels // groups
		group_filter_size = self.info['filters'] // groups

		self.raw_outputs = np.empty((batch_size,) + self.output_shape)

		for group in range(groups):
			channel_range = range(group_channel_size * group, group_channel_size * (group + 1))
			filter_range = range(group_filter_size * group, group_filter_size * (group + 1))

			group_outputs = np.empty((group_channel_size, batch_size) + self.output_shape[:-1] + (group_filter_size,))

			for batch in range(batch_size):
				for filter in filter_range:
					for channel in channel_range:
						group_outputs[channel % group_channel_size, batch, ..., filter % group_filter_size] = functions.conv(
							x[batch, ..., channel],
							self.weights[..., channel % group_channel_size, filter],
							self.info['strides'], self.info['padding'], self.info['dilation_rate']
						)

			self.raw_outputs[..., group_filter_size * group : group_filter_size * (group + 1)] = np.sum(np.array(group_outputs), axis = 0)
		self.raw_outputs += self.biases

		self.outputs = self.activation(self.raw_outputs)

	def backward(self, y: np.ndarray, next_layer: Layer, *args, **kwargs) -> None:
		if self.is_output_layer:
			output_loss = self.outputs - y
		else:
			output_loss = next_layer.input_loss
		raw_output_loss = output_loss * self.activation(self.raw_outputs, True)

		dilated_raw_output_loss = raw_output_loss
		for i in range(len(self.input_shape) - 1):
			idx = np.repeat(np.arange(1, dilated_raw_output_loss.shape[i + 1]), self.info['strides'][i] - 1)
			dilated_raw_output_loss = np.insert(dilated_raw_output_loss, idx, 0, axis = i + 1)

		# print('test', self.inputs.shape, dilated_raw_output_loss.shape)
		print(self.weights.shape, self.info['dilation_rate'])
		weights_shape = np.array(self.weights.shape)
		for i in range(2):
			weights_shape[i] = weights_shape[i] + (self.info['dilation_rate'][i] - 1) * (weights_shape[i] - 1)
		print(weights_shape)
		self.adjustment_weights = np.zeros(self.weights.shape)

		for filter in range(self.info['filters']):																			# TODO add batches
			for channel in range(self.input_shape[2]):
				conv = functions.conv(
					self.inputs[0, :, :, channel],																			# FIXME
					dilated_raw_output_loss[0, :, :, self.info['filters'] * (channel // self.info['groups']) + filter],		# FIXME
					(1, 1),
					'valid',
					(1, 1)
				)
				print('Conv shape', conv.shape)
				conv = conv[::self.info['dilation_rate'][0], ::self.info['dilation_rate'][1]]
				print(conv.shape)
				conv = conv[:self.adjustment_weights.shape[0], :self.adjustment_weights.shape[1]]
				print(conv)
				self.adjustment_weights[:, :, channel % self.info['groups'], filter] += conv
		# print(output_loss[0, :, :, 0])
		# print(self.adjustment_weights[:, :, 0, 0])
		# print(raw_output_loss.shape)

		self.adjustment_biases = np.zeros(self.biases.shape)
		raw_output_loss_sums = np.sum(raw_output_loss, axis = tuple(np.arange(raw_output_loss.ndim - 1)))
		for i in range(raw_output_loss.shape[-1]):
			self.adjustment_biases[i % (self.input_shape[2] // self.info['groups'])] += raw_output_loss_sums[i]				# FIXME

		# print(self.adjustment_biases)

		padding_idx = np.repeat(np.pad(np.array(self.weights.shape[0:2]) - 1, 1), 2).reshape(-1, 2)
		padded_raw_output_loss = np.pad(dilated_raw_output_loss, padding_idx)

		self.input_loss = np.zeros(self.inputs.shape)

		for channel in range(self.input_shape[2]):
			for filter in range(self.info['filters']):
				conv = functions.conv(																						# TODO Batch Inputs, Dilation
					padded_raw_output_loss[0, :, :, self.info['filters'] * (channel // self.info['groups']) + filter],		# FIXME
					np.flip(self.weights[:, :, channel % self.info['groups'], filter]),										# FIXME
					(1, 1),
					'valid',
					(1, 1)
				)
				self.input_loss[0, :conv.shape[0], :conv.shape[1], channel] += conv

		pass	# TODO

	def update(self, learning_rate: float, momentum: float, *args, **kwargs) -> None:
		self.velocity_weights = momentum * self.velocity_weights - (learning_rate * self.adjustment_weights / self.inputs.shape[0])
		self.velocity_biases = momentum * self.velocity_biases - (learning_rate * self.adjustment_biases / self.input_shape[0])

		self.weights = self.weights + self.velocity_weights
		self.biases = self.biases + self.velocity_biases

	def get_weights(self, *args, **kwargs) -> list[np.ndarray]:
		return [self.weights, self.biases]

	def load_weights(self, weights: list[np.ndarray], *args, **kwargs):
		if len(weights) >= 1:
			if weights[0].shape == self.weights.shape:
				self.weights = weights[0]
			else:
				raise ValueError('Mismatched weights ' + str(weights[0].shape) + ' and ' + str(self.weights.shape))
		if len(weights) >= 2:
			if weights[1].shape == self.biases.shape:
				self.biases = weights[1]
			else:
				raise ValueError('Mismatched weights ' + str(weights[1].shape) + ' and ' + str(self.biases.shape))

class Flatten(Layer):
	def __init__(self) -> None:
		self.info =					{}
		self.activation =			af.linear

	def compile(self, input_shape: tuple[int, ...], is_output_layer: bool = False, *args, **kwargs) -> None:
		self.input_shape =			input_shape
		self.output_shape =			(1 if len(input_shape) == 0 else np.prod(input_shape, dtype=int),)

		self.weights =				np.empty(())
		self.biases =				np.empty(())
		self.raw_outputs =			np.empty(())
		self.inputs =				np.empty(())
		self.outputs =				np.empty(())
		self.adjustment_weights =	np.empty(())
		self.adjustment_biases =	np.empty(())
		self.input_loss =			np.empty(())
		self.velocity_weights =		np.empty(())
		self.velocity_biases =		np.empty(())
		self.is_output_layer =		is_output_layer

	def forward(self, x: np.ndarray, *args, **kwargs) -> None:
		self.inputs = x
		self.raw_outputs = x.reshape((-1,) + self.output_shape)
		self.outputs = self.activation(self.raw_outputs)

	def backward(self, y: np.ndarray, next_layer: Layer, *args, **kwargs) -> None:
		if self.is_output_layer:
			self.input_loss = self.outputs - y
		else:
			self.input_loss = next_layer.input_loss
		self.input_loss = self.input_loss.reshape((-1,) + self.input_shape)

	def update(self, *args, **kwargs) -> None:
		return

	def get_weights(self, *args, **kwargs):
		return [self.weights, self.biases]

	def load_weights(self, *args, **kwargs):
		return

class NeuralNetwork(object):
	def __init__(self, input_shape: tuple[int, ...], *layers: Layer) -> None:
		self.input_shape: tuple[int, ...] =		input_shape
		self.layers: List[Layer] =				list(layers)
		self.is_compiled: bool =				False

	def push_layer(self, layer: Layer) -> None:
		self.layers.append(layer)
		self.is_compiled = False

	def pop_layer(self) -> None:
		self.layers.pop()
		self.is_compiled = False

	def compile(self) -> None:
		for i, layer in enumerate(self.layers):
			if i == 0:
				layer.compile(self.input_shape, is_output_layer = (i == len(self.layers) - 1))
			else:
				layer.compile(self.layers[i - 1].output_shape, is_output_layer = (i == len(self.layers) - 1))

		self.is_compiled = True

	def train(
			self, inputs: np.ndarray, outputs: np.ndarray,
			batch_size: Optional[int] = None, learning_rate: float = 0.01, momentum: float = 0.0, epochs: int = 10000,
			validation_split: float = 0.0, validation_data: Optional[tuple[np.ndarray, np.ndarray]] = None, validation_freq: int = 1
		) -> None:

		if not self.is_compiled:
			self.compile()

		if validation_data is not None:
			val_in, val_out = validation_data
			train_in, train_out = inputs, outputs
		elif validation_split == 0.0:
			val_in, val_out = inputs, outputs
			train_in, train_out = inputs, outputs
		else:
			perm = np.random.permutation(inputs.shape[0])
			cutoff = int(validation_split * perm.shape[0])
			val_in, val_out = inputs[perm[:cutoff]], outputs[perm[:cutoff]]
			train_in, train_out = inputs[perm[cutoff:]], outputs[perm[cutoff:]]

		batch_size = batch_size if batch_size else inputs.shape[0]

		print('batch_size:', batch_size, '/', inputs.shape[0])
		print('learning_rate:', learning_rate)
		print('momentum:', momentum)
		print('Training...\n')

		bar_epoch = pb.ProgressBar(max_value = epochs, line_offset = 1, widgets = ['Epoch:        (', pb.Counter(format = '%(value)d of %(max_value)d'), ') ', pb.GranularBar(), ' ', pb.Timer(), ' ', pb.ETA()])
		bar_accuracy = pb.ProgressBar(max_value = 1, widgets = ['Val Accuracy: ', pb.Counter(format = '%(value).4f'), ' ', pb.GranularBar(), ' ', pb.Percentage()])

		for epoch in range(epochs):

			num_batches = train_in.shape[0] // batch_size
			perm = np.random.permutation(train_in.shape[0])

			for iteration in range(num_batches):

				batch_perm = perm[(0 + iteration * batch_size):(batch_size + iteration * batch_size)]

				batch_inputs = train_in[batch_perm]
				batch_outputs = train_out[batch_perm]

				self.layers[0].forward(batch_inputs)
				for i in range(len(self.layers) - 1):
					self.layers[i + 1].forward(self.layers[i].outputs)

				self.layers[-1].backward(batch_outputs, None)
				for i in range(len(self.layers) - 1):
					self.layers[-2 - i].backward(None, self.layers[-1 - i])

				self.layers[0].update(learning_rate, momentum)
				for i in range(len(self.layers) - 1):
					self.layers[i + 1].update(learning_rate, momentum)

				# print(iteration, num_batches)
			bar_epoch.update(epoch + 1)
			if epoch % validation_freq == 0:
				bar_accuracy.update(self.test(val_in, val_out))

		bar_epoch.finish(dirty = True)
		bar_accuracy.finish(dirty = True)

		print('Training complete')

	def think(self, inputs: np.ndarray) -> np.ndarray:
		self.layers[0].forward(inputs)
		for i in range(len(self.layers) - 1):
			self.layers[i + 1].forward(self.layers[i].outputs)

		return self.layers[-1].outputs

	def test(self, inputs: np.ndarray, outputs: np.ndarray, display: bool = False) -> float:
		thought = self.think(inputs)
		y = cross_entropy_to_key(thought)
		y_hat = cross_entropy_to_key(outputs)
		num_correct = np.count_nonzero(y == y_hat)
		if display:
			print('Test Accuracy: ' + str(num_correct) + '/' + str(y_hat.size) + ' = ' + str(num_correct / y_hat.size))
		return num_correct / y_hat.size

	def save_weights(self, file_path: str) -> None:
		glob_weights = {}
		for i in range(len(self.layers)):
			weights = self.layers[i].get_weights()
			for j in range(len(weights)):
				glob_weights['_'.join(str(e) for e in [i, j])] = weights[j]
		np.savez_compressed(file_path, **glob_weights)

	def load_weights(self, file_path: str) -> None:
		with np.load(file_path, allow_pickle = True) as npzfile:
			for i in range(len(self.layers)):
				weights_len = len(self.layers[i].get_weights())
				self.layers[i].load_weights([npzfile['_'.join(str(e) for e in [i, j])] for j in range(weights_len)])


if __name__ == '__main__':
	np.random.seed(0)



	quit()

	# layer.weights = kernel

	# data = np.expand_dims(x_train, -1)[0:batch_size]
	# layer.forward(data)

	# outputs = np.load('keras_outputs.npy', allow_pickle = True)
	# print(layer.outputs.shape, outputs.shape)

	# print(layer.outputs[0, :5, :5, 0])
	# print(outputs[0, :5, :5, 0])
	# print('Abs Max:', np.abs(layer.outputs - outputs).max())
	# idx = np.where(np.abs(layer.outputs - outputs) == np.abs(layer.outputs - outputs).max())
	# print(idx)
	# print(layer.outputs[idx])
	# print(outputs[idx])
	# print('Equal:', np.allclose(layer.outputs, outputs))

	num_inputs = 28 * 28
	num_outputs = 10

	nn = NeuralNetwork((num_inputs,))
	nn.push_layer(Dense(16, af.relu))
	nn.push_layer(Dense(16, af.relu))
	nn.push_layer(Dense(10, af.softmax))

	nn.compile()
	# for i in range(len(nn.layers)):
	# 	nn.layers[i].weights = np.load('keras/weights/0_' + str(i) + '.npy').T
	# 	nn.layers[i].biases = np.load('keras/biases/0_' + str(i) + '.npy').T

	# nn.save_weights('./init_weights.npz')
	# nn.load_weights('./weights.npz')

	path = 'mnist.npz'
	with np.load(path, allow_pickle = True) as f:
		x_train, y_train = f['x_train'], f['y_train']
		x_test, y_test = f['x_test'], f['y_test']

	print(y_train.shape)

	x_train = x_train.reshape((-1, num_inputs)) / 255
	y_train = key_to_one_hot(y_train, np.arange(num_outputs))
	x_test = x_test.reshape((-1, num_inputs)) / 255
	y_test = key_to_one_hot(y_test, np.arange(num_outputs))

	nn.train(x_train, y_train, epochs = 4, batch_size = 64, learning_rate = 0.01, validation_data = (x_train, y_train))

	nn.test(x_test, y_test, display = True)

	# nn.save_weights('./weights.npz')

	print('Finished')