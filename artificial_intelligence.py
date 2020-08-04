import numpy as np
from logic_simulation import logicsim_inputs, logicsim_outputs
# from image_converter import imgconvert_inputs, imgconvert_outputs


def neuron_layer(number_of_inputs_per_neuron, number_of_neurons):
    return 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def train(number_of_training_iterations):
    for iteration in range(number_of_training_iterations):
        for i in range(len(layers)):
            layer_outputs[i + 1] = think(i + 1, training_set_inputs)

        for i in range(len(layers)):
            if i == 0:
                layer_error[len(layers)] = training_set_outputs - layer_outputs[len(layers)]
                for x in range(len(training_set_outputs.flatten())):
                    correct_index = np.where(output_possibilities == training_set_outputs.flatten()[x])[0][0]
                    for y in range(len(output_possibilities.tolist())):
                        layer_error[len(layers)][x, y] = 0 - layer_outputs[len(layers)][x, y]
                    layer_error[len(layers)][x, correct_index] = 1 - layer_outputs[len(layers)][x, correct_index]
            else:
                layer_error[len(layers) - i] = layer_delta[len(layers) - i + 1].dot(layer_weights[len(layers) - i + 1].T)

            layer_delta[len(layers) - i] = layer_error[len(layers) - i] * sigmoid_derivative(layer_outputs[len(layers) - i])

        for i in range(len(layers)):
            if i == 0:
                layer_adjustment[1] = training_set_inputs.T.dot(layer_delta[1])
            else:
                layer_adjustment[i + 1] = layer_outputs[i].T.dot(layer_delta[i + 1])

            layer_weights[i + 1] += layer_adjustment[i + 1]
        if iteration % 1 == 0:
            print('Waiting...  (' + str(iteration + 1) + '/' + str(number_of_training_iterations) + ')')


def think(x, inputs):
    if x == 1:
        output = sigmoid(np.dot(inputs, layer_weights[1]))
    else:
        output = sigmoid(np.dot(layer_outputs[x - 1], layer_weights[x]))
    return output


def simulate(input):
    for i in range(len(layers)):
        layer_outputs[i + 1] = think(i + 1, input)

    sim_confidence = np.amax(layer_outputs[len(layers)])
    sim_output = output_possibilities[np.where(layer_outputs[len(layers)] == np.amax(layer_outputs[len(layers)]))][0]

    return sim_output, sim_confidence


if __name__ == '__main__':
    np.random.seed(1)  # TODO

    output_possibilities = np.array([0, 1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111, 10000, 10001, 10010, 10011, 10100, 10101, 10110, 10111, 11000, 11001, 11010, 11011, 11100, 11101, 11110, 11111,
                                     100000, 100001, 100010, 100011, 100100, 100101, 100110, 100111, 101000, 101001, 101010, 101011, 101100, 101101, 101110, 101111, 110000, 110001, 110010, 110011, 110100, 110101, 110110, 110111, 111000, 111001, 111010, 111011, 111100, 111101, 111110, 111111,
                                     1000000, 1000001, 1000010, 1000011, 1000100, 1000101, 1000110, 1000111, 1001000, 1001001, 1001010, 1001011, 1001100, 1001101, 1001110, 1001111, 1010000, 1010001, 1010010, 1010011, 1010100, 1010101, 1010110, 1010111, 1011000, 1011001, 1011010, 1011011, 1011100, 1011101, 1011110, 1011111,
                                     1100000, 1100001, 1100010, 1100011, 1100100, 1100101, 1100110, 1100111, 1101000, 1101001, 1101010, 1101011, 1101100, 1101101, 1101110, 1101111, 1110000, 1110001, 1110010, 1110011, 1110100, 1110101, 1110110, 1110111, 1111000, 1111001, 1111010, 1111011, 1111100, 1111101, 1111110, 1111111])  # TODO

    training_set_inputs = logicsim_inputs[np.arange(len(logicsim_inputs)) < 61847529062]  # TODO
    training_set_outputs = logicsim_outputs[np.arange(len(logicsim_outputs)) < 61847529062]  # TODO

    layers = {1: [36, 16], 2: [16, 16], 3: [16, 128]}  # TODO
    layer_weights = {}
    layer_outputs = {}
    layer_error = {}
    layer_delta = {}
    layer_adjustment = {}

    for layer in layers:
        layer_weights[layer] = neuron_layer(layers[layer][0], layers[layer][1])

    train(100000)  # TODO

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
    print('Errors: ' + str(error_count) + '  |  Accuracy: ' + '{:.2f}'.format(1 - (error_count / times_simulated)))
