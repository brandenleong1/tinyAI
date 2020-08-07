import numpy as np
import math


def add_and(input1, input2):
    return int(input1) * int(input2)


def add_or(input1, input2):
    return int(input1) + int(input2) - (int(input1) * int(input2))


def add_xor(input1, input2):
    return int(input1) + int(input2) - (2 * int(input1) * int(input2))


def add_not(input1):
    return -1 * (int(input1) - 1)


def add_nand(input1, input2):
    return - 1 * ((int(input1) * int(input2)) - 1)


def add_nor(input1, input2):
    return - 1 * ((int(input1) + int(input2) - (int(input1) * int(input2))) - 1)


def add_xnor(input1, input2):
    return - 1 * ((int(input1) + int(input2) - (2 * int(input1) * int(input2))) - 1)


def simulate(start_from, init):
    print('==================== SIMULATION ====================')
    for iteration in range(start_from, math.floor((2 ** init[2]) // 100000)):
        inputs = np.empty((0, init[2]), dtype='float')
        outputs = np.empty((0, 1), dtype='float')

        for i in range(100000 * iteration, 100000 * (iteration + 1)):
            bin_i = format(i, 'b').zfill(init[2])
            new_row = []
            for x in range(init[2]):
                new_row.append(bin_i[x])
            for x in range(0, init[2]):
                np.append(inputs, [new_row], axis=0)
            np.append(outputs, [[rand_function(i, init[0], init[1], init[2])]], axis=0)
            print(i, bin_i, rand_function(i, init[0], init[1], init[2]))

        np.save('sim_save_in_' + str(iteration), inputs)
        np.save('sim_save_out_' + str(iteration), outputs)

    inputs = np.empty((0, init[2]), dtype='float')
    outputs = np.empty((0, 1), dtype='float')

    for i in range(math.floor((2 ** init[2]) // 100000) * 100000, 2 ** init[2]):
        bin_i = format(i, 'b').zfill(init[2])
        new_row = []
        for x in range(init[2]):
            new_row.append(bin_i[x])
        for x in range(0, init[2]):
            np.append(inputs, [new_row], axis=0)
        np.append(outputs, [[rand_function(i, init[0], init[1], init[2])]], axis=0)
        print(i, bin_i, rand_function(i, init[0], init[1], init[2]))

    np.save('sim_save_in_' + str(math.floor((2 ** init[2]) // 100000)), inputs)
    np.save('sim_save_out_' + str(math.floor((2 ** init[2]) // 100000)), outputs)
    print('====================================================')


def rand_function_initialize(seed):
    np.random.seed(seed)

    PO_len = np.random.randint(2, 20)  # TODO
    num_layers = np.random.randint(2, 20)  # TODO
    layers = {}

    for i in range(num_layers + 1):
        if i == 0:
            layer_temp = []
            for x in range(PO_len):
                layer_temp.append(np.random.randint(0, 7))
            layers[num_layers + 1] = layer_temp
        else:
            num_nodes = np.random.randint(2, len(layers[num_layers - i + 2]) * 2)
            while num_nodes <= len(layers[num_layers - i + 2]):
                num_nodes = np.random.randint(2, len(layers[num_layers - i + 2]) * 2)
            layer_temp = []
            for x in range(num_nodes):
                layer_temp.append(np.random.randint(0, 7))
            layers[num_layers - i + 1] = layer_temp

    PI_len = np.random.randint(PO_len, PO_len * 4)
    return num_layers, layers, PI_len, PO_len


def rand_function(inputs, num_layers, layers, PI_len):
    gate_outputs = [[]]

    bin_i = format(inputs, 'b').zfill(PI_len)

    for i in range(PI_len):
        gate_outputs[0].append(int(bin_i[i]))

    for i in range(num_layers + 1):
        for x in range(len(layers[i + 1])):
            if i == 0:
                input_a = np.random.randint(0, PI_len)
                input_b = np.random.randint(0, PI_len)
                while input_a == input_b:
                    input_b = np.random.randint(0, PI_len)
            else:
                input_a = np.random.randint(0, len(layers[i]))
                input_b = np.random.randint(0, len(layers[i]))
                while input_a == input_b:
                    input_b = np.random.randint(0, len(layers[i]))

        gate_outputs_temp = []
        for x in range(len(layers[i + 1])):
            if layers[i + 1][x] == 0:
                gate_outputs_temp.append(add_not(gate_outputs[i][input_a]))
            elif layers[i + 1][x] == 1:
                gate_outputs_temp.append(gate_outputs[i][input_a])
            elif layers[i + 1][x] == 2:
                gate_outputs_temp.append(add_and(gate_outputs[i][input_a], gate_outputs[i][input_b]))
            elif layers[i + 1][x] == 3:
                gate_outputs_temp.append(add_nand(gate_outputs[i][input_a], gate_outputs[i][input_b]))
            elif layers[i + 1][x] == 4:
                gate_outputs_temp.append(add_or(gate_outputs[i][input_a], gate_outputs[i][input_b]))
            elif layers[i + 1][x] == 5:
                gate_outputs_temp.append(add_nor(gate_outputs[i][input_a], gate_outputs[i][input_b]))
            elif layers[i + 1][x] == 6:
                gate_outputs_temp.append(add_xor(gate_outputs[i][input_a], gate_outputs[i][input_b]))
            elif layers[i + 1][x] == 7:
                gate_outputs_temp.append(add_xnor(gate_outputs[i][input_a], gate_outputs[i][input_b]))

        gate_outputs.append(gate_outputs_temp)

    return int(''.join([str(element) for element in gate_outputs[-1]]))


def output_possibilities_generator(output_length):
    output_possibilities = []
    for i in range(2 ** output_length):
        output_possibilities.append(bin(i))
    return output_possibilities


# ============================================================================================

seed = 1  # TODO

# logicsim_inputs = np.empty((2 ** input_length, input_length), dtype='float')
# logicsim_outputs = np.empty((2 ** input_length, 1), dtype='float')

init = rand_function_initialize(seed)
simulate(0, init)
logicsim_output_possibilities = output_possibilities_generator(init[3])

# logicsim_inputs = np.load('sim_save_in_0.npy')  # TODO
# logicsim_outputs = np.load('sim_save_out_0.npy')  # TODO
# print(np.shape(logicsim_inputs), np.shape(logicsim_outputs))
