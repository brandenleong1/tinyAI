import numpy as np

# ====================================================
# |  XOR.PY                                          |
# |                                                  |
# | Creates an array, {full_in}, for all binary      |
# | numbers with length [inputs] except for the last |
# | [leave_out] numbers. Another array, {full_out},  |
# | with matching indices, tells if there are an odd |
# | or even amount of 1's.                           |
# ====================================================

def xor(inputs, leave_out):
	full_in = []
	full_out = []
	for i in range(2 ** inputs - leave_out):
		num = i
		temp = []
		for j in range(inputs):
			if (num >= (2 ** (inputs - j - 1))):
				temp.append(1)
				num -= 2 ** (inputs - j - 1)
			else:
				temp.append(0)
		full_in.append(temp)
		if (temp.count(1) % 2 == 0):
			full_out.append([0, 1])
		else:
			full_out.append([1, 0])
	
	return np.array(full_in), np.array(full_out)
