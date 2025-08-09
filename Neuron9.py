import numpy as np
np.set_printoptions(linewidth=200)
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
from nnfs.datasets import sine_data

nnfs.init()

import pickle
import copy

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import math

#Input "layer"
class Layer_Input:
    #Forward pass
    def forward(self, inputs, training):
        self.output = inputs


#Dense layer
class Layer_Dense:
    #Dense layer initialization
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        #Initialize weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        #Set regularization strenght
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    #Forward pass
    def forward(self, inputs, training):
        #Calculate output values from inputs, weight and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    #Backward pass
    def backward(self, dvalues):
        #Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        #Gradients on regularization 
        #L1 weights 
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        #L2 weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights 
        #L1 biases 
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        #L2 biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases 


        #Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    #Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases
    
    #Set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


#ReLU activation
class Activation_ReLU:
    #Forward pass
    def forward(self, inputs, training):
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

    #Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs


#Sigmoid activation
class Activation_Sigmoid:
    #Forward pass
    def forward(self, inputs, training):
        #Save input and calculate/save output of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    #Backward pass
    def backward(self, dvalues):
        #Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    #Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1


#Softmax activation
class Activation_Softmax:
    #Forward pass
    def forward(self, inputs, training):
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

    #Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


#Linear Activation
class Activation_Linear:
    #Forward pass
    def forward(self, inputs, training):
        #Just remember the values
        self.inputs = inputs
        self.output = inputs

    #Backward pass
    def backward(self, dvalues):
        #derivative is 1, 1 * dvalues - the chain rule 
        self.dinputs = dvalues.copy()

    #Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs


#Dropout
class Layer_Dropout:
    #Init
    def __init__(self, rate):
        #Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    #Forward pass
    def forward(self, inputs, training):
        #Save input values 
        self.inputs = inputs

        #If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

        #Generate and scale mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        #Applay mask to output values
        self.output = inputs * self.binary_mask

    #Backward pass
    def backward(self, dvalues):
        #Gradient of values
        self.dinputs = dvalues * self.binary_mask


#Common loss class
class Loss:
    #Calculates the data and regularization losses given model output and ground truth values
    def calculate(self, output, y, *, include_regularization=False):
        #Calculate sample losses
        sample_losses = self.forward(output, y)

        #Calculate mean loss
        data_loss = np.mean(sample_losses)

        #Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        #If just data loss - return it
        if not include_regularization:
            return data_loss

        #Return loss
        return data_loss, self.regularization_loss()
    
    #Regularization loss calculation
    def regularization_loss(self):
        #0 by default
        regularization_loss = 0

        #Calculate regularization loss iterate all trainable layers
        for layer in self.trainable_layers:

            #L1 regularization - weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            #L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            #L1 regularization - biases
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            #L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss
    
    #Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers  

    #Calculates accumulated loss
    def calculate_accumulated(self, *, include_regularization=False):
        #Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        #If just data loss - return it
        if not include_regularization:
            return data_loss
        
        #Return the data and regularization losses
        return data_loss, self.regularization_loss()
    
    #Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


#Categorical cross-entropy loss
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
    """not in use
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
    """
    
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


#Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):
    #Forward pass
    def forward(self, y_pred, y_true):
        #Clip data to prevent division by 0
        #Clip both sides to not drag mean towars any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        #Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses
    
    #Backward pass
    def backward(self, dvalues, y_true):
        #Number of samples
        samples = len(dvalues)
        #Number of outputs in every sample 
        #We'll use the first sample to count them
        outputs = len(dvalues[0])

        #Clip data to prevent division by 0
        #Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        #Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs

        #Normalize gradient
        self.dinputs = self.dinputs / samples


#Mean Squared Error loss
class Loss_MeanSquaredError(Loss):
    #Forward pass
    def forward(self, y_pred, y_true):
        #Calculate loss
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)

        #Return loss 
        return sample_losses

    #Backward pass
    def backward(self, dvalues, y_true):
        #Number of samples
        samples = len(dvalues)
        #Number of outputs in every sample
        #We'll use the first sample to count them 
        outputs = len(dvalues[0])

        #Gradient on values 
        self.dinputs = -2 * (y_true - dvalues) / outputs
        #Normalize gradient
        self.dinputs = self.dinputs / samples


#Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss):
    #Forward pass
    def forward(self, y_pred, y_true):
        #Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        #Return loss 
        return sample_losses

    #Backward pass
    def backward(self, dvalues, y_true):
        #Number of samples
        samples = len(dvalues)
        #Number of outputs in every sample
        #We'll use the first sample to count them 
        outputs = len(dvalues[0])

        #Gradient on values 
        self.dinputs = np.sign(y_true - dvalues) / outputs
        #Normalize gradient
        self.dinputs = self.dinputs / samples


#Common accuracy class
class Accuracy:
    #Calculate an accuracy given predictions and ground truth values
    def calculate(self, predictions, y):
        #Get comparison results 
        comparisons = self.compare(predictions, y)

        #Calculate an accuracy
        accuracy = np.mean(comparisons)

        #Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        #Return accuracy
        return accuracy
    
    #Calculates accumulated accuracy
    def calculate_accumulated(self):
        #Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        #Returns the data
        return accuracy
    
    #Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


#Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):
    def __init__(self):
        #Create precision property
        self.precision = None

    #Calculate precision value based on passed in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    #Compares predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


#Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):
    #No initialization is needed
    def init(self, y):
        pass

    #Compare predictions to the ground truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y


#Accuracy calculation for binary classification model
class Accuracy_Binary(Accuracy):
    #No initialization is needed
    def init(self, y):
        pass

    #Compare predictions to the ground truth values
    def compare(self, predictions, y):
        return predictions == y


#SGD optimizer
class Optimizer_SGD:
    #Initialize optimizer - set settings, learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    #Call once before any parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    #Update parameters
    def update_params(self, layer):
        #If we use momentum
        if self.momentum:
            #If layer does not contain momentum arrays, create them filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                #If there is no momentum array for weights the array doesn't exist for biases yet either
                layer.bias_momentums = np.zeros_like(layer.biases)

            #Build weight updates with momentum - take previous updates multiplied by retain factor and update with current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            #Build bias updats
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        #Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        #Update weights and biases using either vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


#Adagrad optimizer
class Optimizer_Adagrad:
    #Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    #Call once before any parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    #Update parameters
    def update_params(self, layer):
        #If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
       
        #Update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        #Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


#RMSprop optimizer
class Optimizer_RMSprop:
    #Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    #Call once before any parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    #Update parameters
    def update_params(self, layer):
        #If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
       
        #Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        #Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


#Adam optimizer
class Optimizer_Adam:
    #Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    #Call once before any parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    #Update parameters
    def update_params(self, layer):
        #If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        #Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
       
        #Get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        #Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        #Get corrected cache
        weight_cache_corrected = layer.weight_cache/ (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache/ (1 - self.beta_2 ** (self.iterations + 1))

        #Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


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
    def graph_create_all_graph(self, model_name, x_start, x_end, y_start, y_end, points_per_axis=401, s=1, cmap='brg', graph_name="", axis_ratio=0, mixing_colors=0):
        #Create data set
        X = np.array([[(x_end - x_start) / (points_per_axis - 1) * i + x_start,
                        (y_end - y_start) / (points_per_axis - 1) * j + y_start]
                        for i in range(points_per_axis)
                        for j in range(points_per_axis)])
        
        #Calculate
        output = model_name.forward(X, training=False)

        if not mixing_colors:
            if isinstance(model_name.layers[-1], Activation_Softmax):
                #Send to draw the graph
                self.graph_create_scatter(X[:,0], X[:,1], np.argmax(output, axis=1), s, cmap, graph_name, axis_ratio)
            elif isinstance(model_name.layers[-1], Activation_Sigmoid):
                #Send to draw the graph
                self.graph_create_scatter(X[:,0], X[:,1], (output < 0.5).ravel(), s, cmap, graph_name, axis_ratio)
        else:
            #Create colors
            #Check if the output is binary
            if len(output[0]) == 1:
                output = np.concatenate([output, 1 - output], axis=1)
            colors = np.array([cm.tab10(i)[:3] for i in range(len(output[0]))])
            colors_expanded = colors[np.newaxis, :, :]
            prob_expanded = output[:, :, np.newaxis]
            result = colors_expanded * prob_expanded
            result_summed = np.sum(result, axis=1)
            result_clipped = np.clip(result_summed, 0, 1)
            self.graph_create_scatter(X[:,0], X[:,1], result_clipped, s, None, graph_name, axis_ratio)
            
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


#Model class
class Model:
    def __init__(self):
        #Create a list of network objects
        self.layers = []
        #Softmax classifier's output object
        self.softmax_classifier_output = None

        #Create lists for saving graph data
        self.all_epochs = []
        self.all_losses = []
        self.all_accuracys = []

    #Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    #Set loss and optimizer
    def set(self, *, loss=None, optimizer=None, accuracy=None):

        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy

    #Finalize the model
    def finalize(self):
        #Create and set the input layer
        self.input_layer = Layer_Input()

        #Count all the objects
        layer_count = len(self.layers)

        #Initialize a list containing trainable layers
        self.trainable_layers = []

        #Iterate the objects
        for i in range(layer_count):
            #If it's the first layer the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            #All layers expect for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            #The last layer - the next onject is the loss
            #Also let's save aside the referance to the last object whose output is the model's output 
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            #If layers contains an attribute calles "weights", it's trainable layer - add it to the list of the trainable layers
            #We don't need to check for biases - checking for the weights is enough
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        #Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        #If output activation is Softmax and loss function is Categorical Cross-Entropy create an object of combined activation and loss function
        #containing faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            #Create an object of combined activation and loss function
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    #Train the model
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):

        #Initalize accuracy object
        self.accuracy.init(y)

        #Default value if batch size is not set
        train_steps=1

        #If there is validation data passed, set default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1

            #For better readability
            X_val, y_val = validation_data

        #Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            #Dividing round down. If there are some remaining data, but not a full batch, this won't include it
            #Add 1 to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                #Dividing round down. If there are some remaining data, but not a full batch, this won't include it
                #Add 1 to include this not full batch
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        #Main training loop
        for epoch in range(1, epochs + 1):

            #Print epoch number
            print(f"epoch: {epoch}")

            #Reset accumulated values in loss and accuracy objects
            self.accuracy.new_pass()
            self.loss.new_pass()

            #Iterate over steps
            for step in range(train_steps):

                #If batch size is not set - train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                #Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                #Perform the forward pass
                output = self.forward(batch_X, training=True)

                #Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                #Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                #Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                #Print a summary and save the data for graph
                if not step % print_every or step == train_steps - 1:
                    print(f"step: {step}, " +
                        f"acc: {accuracy:.3f}, " +
                        f"loss: {loss:.3f} " + 
                        f"(data_loss: {data_loss:.3f}, " +
                        f"reg_loss: {regularization_loss:.3f}) " +
                        f"lr: {self.optimizer.current_learning_rate}")
                    
                    self.all_epochs.append(epoch)
                    self.all_losses.append(loss)
                    self.all_accuracys.append(accuracy)

            #Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()    

            print(f"training, " +
                f"acc: {epoch_accuracy:.3f}, " +
                f"loss: {epoch_loss:.3f} " + 
                f"(data_loss: {epoch_data_loss:.3f}, " +
                f"reg_loss: {epoch_regularization_loss:.3f}) " +
                f"lr: {self.optimizer.current_learning_rate}")
            
        #If there is the validation data
        if validation_data is not None:
            #Evaluate the model:
            self.evaluate(*validation_data, batch_size=batch_size)
            
    #Perform forward pass
    def forward(self, X, training):
        #Call forward method on the input layer this will set the output property that the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        #Call forward method of every object in a chain. Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        #"layer" is now the last object from the list, return its output
        return layer.output
    
    #Perform backward pass
    def backward(self, output, y):

        #If softmax classifier
        if self.softmax_classifier_output is not None:
            #First call backward method on the combined activation/loss this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            #Since we'll not call backward method of the last layer which is Softmax activation as we used combined activation/loss object,
            #let's set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            #Call backward method going through all the objects but last in reverse order passing dinputs as parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        #First call backward method on the loss this will set dinputs property that the last layer will try to access shortly
        self.loss.backward(output, y)

        #Call backward method going through all the objects in reverse order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    #Evaluates the model using passed in dataset
    def evaluate(self, X_val, y_val, *, batch_size=None):
        #Default value if batch size is not set
        validation_steps = 1
        #Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            #Dividing round down. If there are some remaining data, but not a full batch, this won't include it
            #Add 1 to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1


        #Reset accumulated values in loss and accuracy objects
        self.accuracy.new_pass()
        self.loss.new_pass()

        #Iterate over steps
        for step in range(validation_steps):
            #If batch size is not set - train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            #Otherwise slice a batch
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]
            #Perform the forward pass
            output = self.forward(batch_X, training=False)
            #Calculate the loss
            self.loss.calculate(output, batch_y)
            #Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        #Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        #Print a summary
        print(f"validation, " + 
            f"acc: {validation_accuracy:.3f}, " + 
            f"loss: {validation_loss:.3f}")
        
    #Retrieves and returns parameters of trainable layers
    def get_parameters(self):
        #Create a list for parametes
        parameters = []

        #Iterable trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        #Return a list
        return parameters
    
    #Update the model with new parameters
    def set_parameters(self, parameters):
        #Iterate over the parameters and layers and update each layer with each set of parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    #Saves the parameters to a file
    def save_parameters(self, path):
        #Open a file in the binary-write mode and save parameters to it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    #Loads the weights and updates a model instance with them
    def load_parameters(self, path):
        #Open file in the binary-read mode, load weights and update trainable layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    #Saves the model 
    def save(self, path):
        #Make a deep copy of current model instance
        model = copy.deepcopy(self)

        #Reset accumulated values şn loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        #Remove data from input layer and gradients from the loss object
        model.input_layer.__dict__.pop("output", None)
        model.loss.__dict__.pop("dinputs", None)

        #For each layer remove inputs, outputs and dinputs properties
        for layer in model.layers:
            for property in ["inputs", "output", "dinputs", "dweights", "dbiases"]:
                layer.__dict__.pop(property, None)

        #Open a file in the binary-write mode and save the model
        with open(path, "wb") as f:
            pickle.dump(model, f)

    #Loads and returns a model
    @staticmethod
    def load(path):
        #Open file in the binary-read mode, load a model
        with open(path, "rb") as f:
            model = pickle.load(f)

        #Return a model
        return model
    
    #Predicts on the samples
    def predict(self, X, *, batch_size=None):
        #Default value if batch size is not being set
        prediction_steps = 1

        #Calculate number of samples
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            #Dividing round down. If there are some remaining data, but not a full batch, this won't include it
            #Add 1 to include this not full batch
            prediction_steps += 1

        #Model outputs
        output = []

        #Iterate over steps
        for step in range(prediction_steps):
            #If batch size is not set - train using one step and full dataset
            if batch_size is None:
                batch_X = X
            #Otherwise slice a batch
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
            
            #Perform the forward pass
            batch_output = self.forward(batch_X, training=False)

            #Append batch prediction to the list of predictions
            output.append(batch_output)

        #Stack and return results
        return np.vstack(output)

from zipfile import ZipFile
import os
import urllib
import urllib.request
import cv2

URL = "https://nnfs.io/datasets/fashion_mnist_images.zip"
FILE = "fashion_mnist_images.zip"
FOLDER = "fashion_mnist_images"

if not os.path.isfile(FILE):
    print(f"Downloading {URL} and saving as {FILE}...")
    urllib.request.urlretrieve(URL, FILE)

if not os.path.isdir(FOLDER):
    print("Unzipping images...")
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)

    print("Done!")


#Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
    #Scan all the directories and create a list of lables
    labels = os.listdir(os.path.join(path, dataset))

    #Create lists for samples and labels
    X = []
    y = []

    #For each label folder
    for label in labels:
        #And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            #Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            #And append it and a label to the lists
            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')

#MNIST dataset (train + test)
def create_data_mnist(path):
    #Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    #And return all the data
    return X, y, X_test, y_test


#Create dataset
X, y, X_test, y_test = create_data_mnist(FOLDER)

#Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

#Scale and reshape samples
X = ((X.reshape(X.shape[0], -1)).astype(np.float32) - 127.5) / 127.5
X_test = ((X_test.reshape(X_test.shape[0], -1)).astype(np.float32) - 127.5) / 127.5

#Instantiate the model
model = Model()

#Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

#Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

#Finalize the model
model.finalize()

#Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)

#Save the model
model.save('fashion_mnist.model')

#Label index to label name relation
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

#Read an image
path = os.path.join("images", "dress.png")
image_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

#Resize to the same size as Fashion MNIST images
image_data = cv2.resize(image_data, (28, 28))

#Invert image colors
image_data = 255 - image_data

#Reshape and scale pixel data
image_data = ((image_data.reshape(1, -1)).astype(np.float32) - 127.5) / 127.5

#Prediction on this image
predictions = model.predict(image_data)

#Get prediction instead of confidance levels
predictions = model.output_layer_activation.predictions(predictions)

#Get label name from label index
predictions = fashion_mnist_labels[predictions[0]]

print(predictions)