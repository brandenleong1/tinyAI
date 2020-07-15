import numpy as np
# from logic_simulation import logicsim_inputs, logicsim_outputs
from image_converter import imgconvert_inputs, imgconvert_outputs


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

    output_possibilities = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # TODO

    training_set_inputs = imgconvert_inputs[np.arange(len(imgconvert_inputs)) > 0]  # TODO
    training_set_outputs = imgconvert_outputs[np.arange(len(imgconvert_outputs)) > 0]  # TODO

    layers = {1: [1600, 16], 2: [16, 16], 3: [16, 16], 4: [16, 11]}  # TODO
    layer_weights = {}
    layer_outputs = {}
    layer_error = {}
    layer_delta = {}
    layer_adjustment = {}

    for layer in layers:
        layer_weights[layer] = neuron_layer(layers[layer][0], layers[layer][1])

    train(10000)  # TODO

    error_count = 0
    times_simulated = 0

    for i in imgconvert_inputs:
        if not any((training_set_inputs[:] == i).all(1)):
            times_simulated += 1
            simulate(i)
            sim_out = simulate(i)[0]
            sim_con = simulate(i)[1]
            if sim_out != imgconvert_outputs[imgconvert_inputs.tolist().index(i.tolist())]:
                print(sim_out, imgconvert_outputs[imgconvert_inputs.tolist().index(i.tolist())])
                error_count += 1
    print('Errors: ' + str(error_count) + '  |  Accuracy: ' + '{:.2f}'.format(1 - (error_count / times_simulated)))
