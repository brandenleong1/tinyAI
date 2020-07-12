import numpy as np


def add_and(input1, input2):
    return int(input1) * int(input2)


def add_or(input1, input2):
    return int(input1) + int(input2) - (int(input1) * int(input2))


def add_xor(input1, input2):
    return int(input1) + int(input2) - (2 * int(input1) * int(input2))


def function(i1, i2, i3, i4, i5, i6):
    return add_and(add_and(add_xor(i1, i2), add_xor(i3, i4)), add_xor(i5, i6))                  # TODO


input_length = 6                                                                                # TODO

inputs = np.empty((2 ** input_length, input_length), dtype='int64')
outputs = np.empty((2 ** input_length, 1), dtype='int64')

print('==================== SIMULATION ====================')
for i in range(2 ** input_length):
    bin_i = format(i, 'b').zfill(input_length)
    for x in range(0, input_length):
        inputs[i, x] = bin_i[x]
    outputs[i, 0] = function(bin_i[0], bin_i[1], bin_i[2], bin_i[3], bin_i[4], bin_i[5])        # TODO
    print(i, bin_i, function(bin_i[0], bin_i[1], bin_i[2], bin_i[3], bin_i[4], bin_i[5]))          # TODO
print('====================================================')
