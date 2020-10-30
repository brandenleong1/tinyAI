import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.piecewise(x, [x < 0, x >= 0], [0, 1])


class Layer:
    def __init__(self, dim1, dim2, type, type_derivative, isOutputLayer=False):
        self.weights = np.random.random((dim1, dim2))
        self.biases = np.random.random((dim1, 1))
        self.raw_outputs = None
        self.outputs = None
        self.adjustment = None
        self.type = type
        self.type_derivative = type_derivative
        self.isOutputLayer = isOutputLayer

    def forward(self, x):
        self.raw_outputs = np.dot(self.weights, x) + self.biases
        self.outputs = self.type(self.raw_outputs)

    def backward(self, y, next_layer):
        if self.isOutputLayer:
            self.adjustment = (self.outputs - y) * self.type_derivative(self.raw_outputs)
        else:
            self.adjustment = np.dot(next_layer.weights.T, next_layer.adjustment) * self.type_derivative(self.raw_outputs)

    def update_weights(self, learning_rate, x):
        self.weights = self.weights - (learning_rate * np.dot(self.adjustment, x.T))
        self.biases = self.biases - (learning_rate * self.adjustment)


class NeuralNetwork:
    def __init__(self, nodes, type='relu'):
        if type == 'sigmoid':
            self.type = sigmoid
            self.type_derivative = sigmoid_derivative
        elif type == 'relu':
            self.type = relu
            self.type_derivative = relu_derivative
        elif type == 'tanh':
            self.type = tanh
            self.type_derivative = tanh_derivative

        self.layers = []
        for i in range(1, len(nodes) - 1):
            self.layers.append(Layer(nodes[i], nodes[i - 1], self.type, self.type_derivative))
        self.layers.append(Layer(nodes[len(nodes) - 1], nodes[len(nodes) - 2], self.type, self.type_derivative, True))

    def train(self, inputs, outputs, learning_rate=0.1, epochs=10000, debug=False, w=False, b=False, r=False, o=False, a=False):
        print('Training...')
        for iteration1 in range(int(np.floor(np.sqrt(epochs)))):
            for set_number in range(len(inputs)):
                for iteration2 in range(int(np.floor(np.sqrt(epochs)))):

                    self.layers[0].forward(np.array([inputs[set_number]]).T)
                    for i in range(len(self.layers) - 1):
                        self.layers[i + 1].forward(self.layers[i].outputs)

                    self.layers[-1].backward(np.array([outputs[set_number]]).T, None)
                    for i in range(len(self.layers) - 1):
                        self.layers[-2 - i].backward(None, self.layers[-1 - i])

                    if debug:
                        for i in range(len(self.layers)):
                            if w or (not b and not r and not o and not a):
                                print('L' + str(i + 1) + '.weights\n', self.layers[i].weights, '\n')
                            if b or (not w and not r and not o and not a):
                                print('L' + str(i + 1) + '.biases\n', self.layers[i].biases, '\n')
                            if r or (not w and not b and not o and not a):
                                print('L' + str(i + 1) + '.raw_outputs\n', self.layers[i].raw_outputs, '\n')
                            if o or (not w and not b and not r and not a):
                                print('L' + str(i + 1) + '.outputs\n', self.layers[i].outputs, '\n')
                            if a or (not w and not b and not r and not o):
                                print('L' + str(i + 1) + '.adjustment\n', self.layers[i].adjustment, '\n')

                    self.layers[0].update_weights(learning_rate, np.array([inputs[set_number]]).T)
                    for i in range(len(self.layers) - 1):
                        self.layers[i + 1].update_weights(learning_rate, self.layers[i].outputs)

        print('Training complete.')

    def think(self, inputs):
        self.layers[0].forward(np.array([inputs]).T)
        for i in range(len(self.layers) - 1):
            self.layers[i + 1].forward(self.layers[i].outputs)

        return self.layers[-1].outputs

    def test(self, inputs, outputs):
        correct = 0
        for set_number in range(len(inputs)):
            thought = self.think(inputs[set_number])
            if np.array_equal(np.where(thought == np.amax(thought), 1, 0), np.array([outputs[set_number]]).T):
                correct += 1
                print('Correct ', outputs[set_number], np.amax(thought))
            else:
                print('Wrong   ', np.where(thought == np.amax(thought), 1, 0).T[0], '', outputs[set_number], np.amax(thought))
        print('\n' + str(correct) + '/' + str(len(inputs)) + '  =  ' + str(correct / len(inputs)))


if __name__ == '__main__':
    a = NeuralNetwork([inputs, 10, 9, 2], type='tanh')
    a.train(full_in, full_out, learning_rate=0.3, epochs=10000)

    test_in = full_in
    test_out = full_out
    a.test(test_in, test_out)
