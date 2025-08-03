import numpy as np

inputs = np.array([[1.0, 5.0, 4.0, 1.7],
                   [1.0, 2.0, 3.0, 4.0]])

weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.7, -0.2, 0.6, 0.4],
                    [0.3, 0.2, 0.1, 0.9]])

biases = np.array([2.0, 7.0, 10.0])

weights2 = np.array([[0.2, 0.8, -0.5],
                    [0.7, -0.2, 0.6],
                    [0.3, 0.2, 0.1],
                    [-0.5, 0.4, -0.8],
                    [0.2, -0.2, -0.7]])

biases2= np.array([5.0, 3.7, 9.9, 1.1, 5.6])

layer1_outputs = np.dot(inputs, weights.T) + biases

layer2_outputs = np.dot(layer1_outputs, weights2.T) + biases2

print(layer2_outputs)