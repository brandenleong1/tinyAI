import numpy as np
from simulation import inputs, outputs


def neuron_layer(number_of_inputs_per_neuron, number_of_neurons):
    return 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def train(number_of_training_iterations):
    for iteration in range(number_of_training_iterations):
        if iteration % 2000 == 0:
            print('Waiting...  (' + str(iteration) + '/' + str(number_of_training_iterations) + ')')
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

    output_possibilities = np.array([0, 1])  # TODO

    training_set_inputs = inputs[np.arange(len(inputs)) < 42]  # TODO
    training_set_outputs = outputs[np.arange(len(outputs)) < 42]  # TODO

    layers = {1: [6, 10], 2: [10, 10], 3: [10, 2]}  # TODO
    layer_weights = {}
    layer_outputs = {}
    layer_error = {}
    layer_delta = {}
    layer_adjustment = {}

    for layer in layers:
        layer_weights[layer] = neuron_layer(layers[layer][0], layers[layer][1])

    train(10000)  # TODO

    test = inputs[42]  # TODO

    print('Output: ' + str(simulate(test)[0]))
    print('Confidence: ' + str(simulate(test)[1]))
