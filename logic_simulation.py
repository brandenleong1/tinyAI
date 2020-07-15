import numpy as np


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


def function(i1, i2, i3, i4, i5):  # TODO
    return ((add_nand(add_nand(i1, i3), add_nand(i2, add_nand(i3, i4)))) * 10) + (add_nand(add_nand(i2, add_nand(i3, i4)), add_nand(i5, add_nand(i3, i4))))  # TODO


input_length = 5  # TODO

logicsim_inputs = np.empty((2 ** input_length, input_length), dtype='int64')
logicsim_outputs = np.empty((2 ** input_length, 1), dtype='int64')

print('==================== SIMULATION ====================')
for i in range(2 ** input_length):
    bin_i = format(i, 'b').zfill(input_length)
    for x in range(0, input_length):
        logicsim_inputs[i, x] = bin_i[x]
    logicsim_outputs[i, 0] = function(bin_i[0], bin_i[1], bin_i[2], bin_i[3], bin_i[4])  # TODO
    print(i, bin_i, function(bin_i[0], bin_i[1], bin_i[2], bin_i[3], bin_i[4]))  # TODO
print('====================================================')
