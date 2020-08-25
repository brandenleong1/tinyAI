import numpy as np
from logic_simulation import logicsim_inputs, logicsim_outputs, logicsim_output_possibilities, init
# from image_converter import imgconvert_inputs, imgconvert_outputs


def weight_init(number_of_inputs_per_neuron, number_of_neurons):
    return np.random.random((number_of_neurons, number_of_inputs_per_neuron))


def bias_init(number_of_neurons):
    return np.random.random((number_of_neurons, 1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.piecewise(x, [x < 0, x >= 0], [0, 1])


def train(number_of_training_iterations):
    for iteration in range(number_of_training_iterations):
        for i in range(len(layers)):
            layer_avg_adjustment[i + 1] = np.zeros(np.shape(layer_weights[i + 1]))

        for input_set in range(len(training_set_inputs)):
            for i in range(len(layers)):
                layer_outputs[i + 1] = think(i + 1, np.array([training_set_inputs[input_set].tolist()]).T)

            for i in range(len(layers)):
                if i == 0:
                    layer_error[len(layers)] = np.zeros((len(layer_outputs[len(layers)]), 1))
                    correct_index = np.where(output_possibilities == training_set_outputs[input_set][0])[0][0]
                    for x in range(len(layer_outputs[len(layers)])):
                        layer_error[len(layers)][x, 0] = 0 - layer_outputs[len(layers)][x, 0]
                    layer_error[len(layers)][correct_index, 0] = 1 - layer_outputs[len(layers)][correct_index, 0]
                else:
                    layer_error[len(layers) - i] = np.dot(np.ones((1, len(layer_outputs[len(layers) - i + 1]))), layer_weights_adjustment[len(layers) - i + 1]).T

                layer_error_identity[len(layers) - i] = np.zeros((len(layer_error[len(layers) - i]), len(layer_error[len(layers) - i])))
                for x in range(len(layer_outputs[len(layers) - i])):
                    layer_error_identity[len(layers) - i][x][x] = layer_error[len(layers) - i][x][0]
                if i == 0:
                    layer_weights_adjustment[len(layers) - i] = np.dot(layer_error_identity[len(layers) - i], np.dot(sigmoid_derivative(layer_outputs[len(layers)- i]), layer_outputs[len(layers) - i - 1].T))
                    layer_biases_adjustment[len(layers) - i] = np.dot(layer_error_identity[len(layers) - i], sigmoid_derivative(layer_outputs[len(layers) - i]))
                elif i == len(layers) - 1:
                    layer_weights_adjustment[len(layers) - i] = np.dot(layer_error_identity[len(layers) - i], np.dot(sigmoid_derivative(layer_outputs[len(layers) - i]), np.array([training_set_inputs[input_set].tolist()]))) # relu
                    layer_biases_adjustment[len(layers) - i] = np.dot(layer_error_identity[len(layers) - i], sigmoid_derivative(layer_outputs[len(layers) - i])) # relu
                else:
                    layer_weights_adjustment[len(layers) - i] = np.dot(layer_error_identity[len(layers) - i], np.dot(sigmoid_derivative(layer_outputs[len(layers) - i]), layer_outputs[len(layers) - i - 1].T)) # relu
                    layer_biases_adjustment[len(layers) - i] = np.dot(layer_error_identity[len(layers) - i], sigmoid_derivative(layer_outputs[len(layers) - i])) # relu

        for i in range(len(layers)):
            layer_weights[i + 1] += (layer_avg_adjustment[i + 1] * (1 / len(training_set_inputs)))
            layer_biases[i + 1] += layer_biases_adjustment[i + 1]

        if iteration % 1 == 0:
            print('Waiting...  (' + str(iteration + 1) + '/' + str(number_of_training_iterations) + ')')
        if iteration + 1 == number_of_training_iterations:
            print('Simulating tests...')


def think(x, inputs):
    if x == 1:
        output = sigmoid(np.dot(layer_weights[1], inputs) + layer_biases[1]) # relu
    elif x == len(layers):
        output = sigmoid(np.dot(layer_weights[x], layer_outputs[x - 1]) + layer_biases[x])
    else:
        output = sigmoid(np.dot(layer_weights[x], layer_outputs[x - 1]) + layer_biases[x])
    return output


def simulate(input):
    for i in range(len(layers)):
        layer_outputs[i + 1] = think(i + 1, np.array([input.tolist()]).T)

    sim_confidence = np.amax(layer_outputs[len(layers)])
    sim_output = output_possibilities[np.where(layer_outputs[len(layers)] == np.amax(layer_outputs[len(layers)]))[0]]

    return sim_output, sim_confidence


if __name__ == '__main__':
    output_possibilities = np.array(logicsim_output_possibilities)  # TODO

    training_set_inputs = logicsim_inputs[np.arange(len(logicsim_inputs)) < (len(logicsim_inputs) * 0.9)]  # TODO
    training_set_outputs = logicsim_outputs[np.arange(len(logicsim_outputs)) < (len(logicsim_outputs) * 0.9)]  # TODO

    layers = {1: [init[2], 16], 2: [16, 16], 3: [16, len(output_possibilities.tolist())]}  # TODO
    layer_weights = {}
    layer_biases = {}
    layer_outputs = {}
    layer_error = {}
    layer_error_identity = {}
    layer_weights_adjustment = {}
    layer_biases_adjustment = {}
    layer_avg_adjustment = {}

    for layer in layers:
        layer_weights[layer] = weight_init(layers[layer][0], layers[layer][1])
        layer_biases[layer] = bias_init(layers[layer][1])

    train(1)  # TODO

    error_count = 0
    times_simulated = 0

    for i in logicsim_inputs:
        if not any((training_set_inputs[:] == i).all(1)):
            times_simulated += 1
            sim_out, sim_conf = simulate(i)
            if sim_out != logicsim_outputs[logicsim_inputs.tolist().index(i.tolist())]:
                print(sim_out, sim_conf, logicsim_outputs[logicsim_inputs.tolist().index(i.tolist())])
                error_count += 1
                print('Error count: ' + str(error_count))
            else:
                print('Success! ' + str(sim_out) + '  ' + str(sim_conf))
    print('Errors: ' + str(error_count) + '/' + str(times_simulated) + '  |  Accuracy: ' + '{:.2f}'.format(1 - (error_count / times_simulated)))
