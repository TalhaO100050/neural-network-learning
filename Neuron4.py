import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data

nnfs.init()

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import math

#Dense layer
class Layer_Dense:
    #Dense layer initialization
    def __init__(self, n_inputs, n_neurons):
        #Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    #Forward pass
    def forward(self, inputs):
        #Calculate output values from inputs, weight and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    #Backward pass
    def backward(self, dvalues):
        #Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


#ReLU activation
class Activation_ReLU:
    #Forward pass
    def forward(self, inputs):
        #Calculate output values from inputs
        self.output = np.maximum(0, inputs)

        #Remember the input values
        self.inputs = inputs

    #Backward pass
    def backward(self, dvalues):
        #Since we need to modify the original variable, let's make a copy of the values first
        self.dinputs = dvalues.copy()

        #Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


#Softmax activation
class Activation_Softmax:
    #Forward pass
    def forward(self, inputs):
        #Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        #Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    #Backward pass
    def backward(self, dvalues):
        #Create uninitilazed array
        self.dinputs = np.empty_like(dvalues)

        #Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #Flatten output array
            single_output = single_output.reshape(-1,1)
            #Calculate Jacobian matrix of the output and 
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #Calculate sample-wise gradient and add it to the array of sample gradients 
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        

#Common loss class
class Loss:
    #Calculates the data and regularization losses given model output and ground truth values
    def calculate(self, output, y):
        #Calculate sample losses
        sample_losses = self.forward(output, y)

        #Calculate mean loss
        data_loss = np.mean(sample_losses)

        #Return loss
        return data_loss


#Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    #Forward pass
    def forward(self, y_pred, y_true):
        #Number of samples in a batch
        samples = len(y_pred)

        #Clip data to prevent division by 0
        #Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        #Probabilities for target values - only if categorical lables
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        #Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        #Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    #Backward pass
    def backward(self, dvalues, y_true):
        #Number of samples
        samples = len(dvalues)
        #Number of labels in every sample we'll use the first sample to count them
        labels = len(dvalues[0])

        #If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        #Calculate gradient
        self.dinputs = -y_true / dvalues
        #Normalize gradient
        self.dinputs = self.dinputs/ samples


#Softmax classifier - combined Softmax activation and cross-entropy loss for fast backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    #Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    #Forward pass
    def forward(self, inputs, y_true):
        #Output layer's activation function
        self.activation.forward(inputs)
        #Set the output
        self.output = self.activation.output
        #Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    #Backward pass
    def backward(self, dvalues, y_true):
        #Number of samples
        samples = len(dvalues)
        #If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        #Copy so we can safely modify
        self.dinputs = dvalues.copy()
        #Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        #Normalize gradient
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:
    #Initialize optimizer - set settings, learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    #Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


#Create graphs
class Graph_maker:
    def __init__(self):
        self.graph_size = 0
        self.graphs = []

    #Change graph size
    def set_graph_size(self, graph_size):
        self.graph_size = graph_size

    #Delete saved graphs
    def clear_graphs(self):
        self.graphs = []

    #Basic graph - scatter graph
    def graph_create_scatter(self, X, y, c=0, s=15, cmap='brg', graph_name="", axis_ratio=0):
        self.graphs.append([0, X.copy(), y.copy(), c.copy(), s, cmap, graph_name, axis_ratio])

    #Basic graph - plot graph
    def graph_create_plot(self, X, Y, graph_name="", axis_ratio=0):
        self.graphs.append([1, X.copy(), Y.copy(), graph_name, axis_ratio])

    #Custom graph - all data
    def graph_create_all_graph(self, x_start, x_end, y_start, y_end, points_per_axis=401, s=1, cmap='brg', graph_name="", axis_ratio=0, mixing_colors=0):
        #Create data set
        X = np.array([[(x_end - x_start) / (points_per_axis - 1) * i + x_start,
                        (y_end - y_start) / (points_per_axis - 1) * j + y_start]
                        for i in range(points_per_axis)
                        for j in range(points_per_axis)])
        
        #Calculate
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation_softmax = Activation_Softmax()
        activation_softmax.forward(dense2.output)

        if not mixing_colors:
            #Send to draw the graph
            self.graph_create_scatter(X[:,0], X[:,1], np.argmax(activation_softmax.output, axis=1), s, cmap, graph_name, axis_ratio)
        else:
            #Create colors
            colors = np.array([cm.tab10(i)[:3] for i in range(len(activation_softmax.output[0]))])
            colors_exp = colors[np.newaxis, :, :]
            prob_exp = activation_softmax.output[:, :, np.newaxis]
            result = colors_exp * prob_exp
            summed_result = np.sum(result, axis=1)
            summed_result = np.clip(summed_result, 0, 1)
            self.graph_create_scatter(X[:,0], X[:,1], summed_result, s, None, graph_name, axis_ratio)
            
    #Show graph
    def graph_show(self):
        graph_number = len(self.graphs)

        #Calculate shape of the graph
        if self.graph_size:
            graph_row = self.graph_size[1]
            graph_column = self.graph_size[0]
        else:
            if graph_number <= 2:
                graph_row = 1
                graph_column = math.ceil(graph_number / graph_row)
            elif graph_number <= 6:
                graph_row = 2
                graph_column = math.ceil(graph_number / graph_row)
            else:
                graph_row = 3
                graph_column = math.ceil(graph_number / graph_row)

        #Window shape
        plt.figure(figsize=(graph_column * 5, graph_row * 5))

        #Draw graphs
        for i in range(len(self.graphs)):
            plt.subplot(graph_row, graph_column, i + 1)

            #Graph data
            graph_data = self.graphs[i]

            #Make graph
            if graph_data[0] == 0: #Scatter graph

                if isinstance(graph_data[3], np.ndarray): #graph_data[3] = c
                    if graph_data[5] != None:
                        plt.scatter(graph_data[1], graph_data[2], c=graph_data[3], s=graph_data[4], cmap=graph_data[5])
                    else:
                        plt.scatter(graph_data[1], graph_data[2], c=graph_data[3], s=graph_data[4])
                else:
                    plt.scatter(graph_data[1], graph_data[2], s=graph_data[4])
                
                #Set name
                plt.title(graph_data[6])

                #Set ratio
                if graph_data[7] == 1: #graph_data[7] = axis_ratio
                    plt.axis('equal')

            elif graph_data[0] == 1: #Plot graph

                plt.plot(graph_data[1], graph_data[2])

                #Set name
                plt.title(graph_data[3])

                #Set ratio
                if graph_data[4]: #graph_data[4] = axis_ratio
                    plt.axis('equal')

        plt.show()



#Create dataset
X, y = spiral_data(samples=100, classes=3)
#X, y = vertical_data(samples=100, classes=3)

#Neuron numbers
layer1 = 2
layer2 = 64
layer3 = 3


#Create layers
dense1 = Layer_Dense(layer1,layer2)
dense2 = Layer_Dense(layer2,layer3)

#Create activation functions
activation1 = Activation_ReLU()

#Create loss function
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#Create optimizer
optimizer = Optimizer_SGD()

#Make lists for graphs
all_loss = []
all_accuracy = []
all_epoch = []

#Train in loop
for epoch in range(10001):

    #Calculate
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    loss = loss_activation.forward(dense2.output, y)

    #Calculate accuracy from output of activation3 and targets calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch} ' + f'acc: {accuracy:.3} ' + f'loss: {loss:.3}')
        all_loss.append(loss)
        all_accuracy.append(accuracy)
        all_epoch.append(epoch)

    #Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    #Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

#Create graphs
graph_maker = Graph_maker()

#Example, guess, loss and accuracy graphs
graph_maker.graph_create_scatter(X[:,0], X[:,1], c=y, graph_name="Gerçek veri")
graph_maker.graph_create_scatter(X[:,0], X[:,1], c=np.argmax(loss_activation.output, axis=1), graph_name="Tahmin")
graph_maker.graph_create_plot(all_epoch, all_loss, graph_name="Loss graph")
graph_maker.graph_create_plot(all_epoch, all_accuracy, graph_name="Accuracy graph")
graph_maker.graph_show()
graph_maker.clear_graphs()

#Graph for all datas
x_start_end = np.array([-1, 1])
y_start_end = np.array([-1, 1])
big_graph_ratio = 4
x_start_end_big = x_start_end * big_graph_ratio
y_start_end_big = y_start_end * big_graph_ratio

graph_maker.graph_create_all_graph(x_start_end[0], x_start_end[1], y_start_end[0], y_start_end[1], graph_name="All possible data", axis_ratio=1)
graph_maker.graph_create_all_graph(x_start_end_big[0], x_start_end_big[1], y_start_end_big[0], y_start_end_big[1], graph_name="All possible data big", axis_ratio=1)
graph_maker.graph_create_all_graph(x_start_end[0], x_start_end[1], y_start_end[0], y_start_end[1], graph_name="All possible data all accuracys", axis_ratio=1, mixing_colors=1)
graph_maker.graph_create_all_graph(x_start_end_big[0], x_start_end_big[1], y_start_end_big[0], y_start_end_big[1], graph_name="All possible data big all accuracys", axis_ratio=1, mixing_colors=1)
graph_maker.graph_show()