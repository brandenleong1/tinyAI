import numpy as np

def conv_shape(
	input_shape: tuple[int, ...],
	filter_shape: tuple[int, ...],
	strides: tuple[int, ...],
	padding: tuple[tuple[int, int], ...] | str,
	dilation: tuple[int, ...]
) -> tuple[int, ...]:

	num_dims = len(input_shape)
	input_shape = np.array(input_shape)
	filter_shape = np.array(filter_shape)
	filter_shape = (filter_shape - 1) * (np.array(dilation) - 1) + filter_shape

	if type(padding) == tuple:
		padded_input_shape = input_shape + np.sum(padding, axis = 1)
	else:
		if padding.lower() == 'valid':
			padded_input_shape = input_shape
		elif padding.lower() == 'same':
			pad_along_axis = np.empty(num_dims)
			for i in range(num_dims):
				if input_shape[i] % strides[i] == 0:
					pad = max(filter_shape[i] - strides[i], 0)
				else:
					pad = max(filter_shape[i] - (input_shape[i] % strides[i]), 0)
				pad_along_axis[i] = pad

			padded_input_shape = input_shape + pad_along_axis

	filter_pad = filter_shape // 2
	padded_input_shape += filter_pad * 2

	valid_shape = padded_input_shape - filter_shape + 1

	valid_shape -= filter_pad * 2
	valid_shape = np.ceil(valid_shape / strides).astype('int')

	return tuple(valid_shape)

def conv(
	input: np.ndarray,
	filter: np.ndarray,
	strides: tuple[int, ...],
	padding: tuple[tuple[int, int], ...] | str,
	dilation: tuple[int, ...]
) -> np.ndarray:

	filter = np.flip(filter)
	num_dims = len(input.shape)

	for i in range(num_dims):
		idx = np.repeat(np.arange(1, filter.shape[i]), dilation[i] - 1)
		filter = np.insert(filter, idx, 0, axis = i)

	if type(padding) == tuple:
		padded_input = np.pad(input, padding)
	else:
		if padding.lower() == 'valid':
			padded_input = input
		elif padding.lower() == 'same':
			pad_along_axis = []
			for i in range(num_dims):
				if input.shape[i] % strides[i] == 0:
					pad = max(filter.shape[i] - strides[i], 0)
				else:
					pad = max(filter.shape[i] - (input.shape[i] % strides[i]), 0)
				pad_along_axis.append((pad // 2, pad - (pad // 2)))

			padded_input = np.pad(input, pad_along_axis)

	filter_pad = np.array(filter.shape) // 2
	padded_input = np.pad(padded_input, np.repeat(filter_pad.reshape(-1, 1), 2, axis = 1))

	input_shape = np.array(padded_input.shape)
	filter_shape = np.array(filter.shape)

	output_shape = input_shape + filter_shape - 1
	output = np.fft.ifftn(np.fft.fftn(padded_input, output_shape) * np.fft.fftn(filter, output_shape)).real

	valid_shape = input_shape - filter_shape + 1

	start = (output_shape - valid_shape) // 2
	end = start + valid_shape

	output = output[*[slice(*i) for i in np.array([start, end]).T]]
	output = output[*[slice(*i) for i in np.array([filter_pad, -filter_pad, strides]).T]]

	return output

def conv_wrapper(inputs, filters, strides, padding, dilation, groups):
	batch_size = inputs.shape[0]
	channels = inputs.shape[-1]
	filter_size = filters.shape[-1]

	group_channel_size = channels // groups
	group_filter_size = filter_size // groups

	output_shape = conv_shape(inputs.shape[1:-1], filters.shape[:-2], strides, padding, dilation)

	outputs = np.empty((batch_size,) + output_shape + (filter_size,))

	for group in range(groups):
		channel_range = range(group_channel_size * group, group_channel_size * (group + 1))
		filter_range = range(group_filter_size * group, group_filter_size * (group + 1))

		group_outputs = np.empty((group_channel_size, batch_size) + output_shape + (group_filter_size,))

		for batch in range(batch_size):
			for filter in filter_range:
				for channel in channel_range:
					group_outputs[channel % group_channel_size, batch, ..., filter % group_filter_size] = conv(inputs[batch, ..., channel], filters[..., channel % group_channel_size, filter], strides, padding, dilation)

		outputs[..., group_filter_size * group : group_filter_size * (group + 1)] = np.sum(np.array(group_outputs), axis = 0)

	return outputs