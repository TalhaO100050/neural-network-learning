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
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        #Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        #Set regularization strenght
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

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


#Sigmoid activation
class Activation_Sigmoid:
    #Forward pass
    def forward(self, inputs):
        #Save input and calculate/save output of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    #Backward pass
    def backward(self, dvalues):
        #Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output


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
        

#Dropout
class Layer_Dropout:
    #Inıt
    def __init__(self, rate):
        #Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    #Forward pass
    def forward(self, inputs):
        #Save input values 
        self.inputs = inputs
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
    def calculate(self, output, y):
        #Calculate sample losses
        sample_losses = self.forward(output, y)

        #Calculate mean loss
        data_loss = np.mean(sample_losses)

        #Return loss
        return data_loss
    
    #Regularization loss calculation
    def regularization_loss(self, layer):
        #0 by default
        regularization_loss = 0

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
        sample_losses = np.mean(sample_losses)

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
        activation_sigmoid = Activation_Sigmoid()
        activation_sigmoid.forward(dense2.output)

        if not mixing_colors:
            #Send to draw the graph
            self.graph_create_scatter(X[:,0], X[:,1], (activation_sigmoid.output >= 0.5).ravel(), s, cmap, graph_name, axis_ratio)
        else:
            #Create colors
            gray_colors = np.repeat(activation_sigmoid.output, 3, axis=1) 
            self.graph_create_scatter(X[:,0], X[:,1], gray_colors, s, None, graph_name, axis_ratio)
            
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
X, y = spiral_data(samples=1000, classes=2)

y = y.reshape(-1, 1)

#Neuron numbers
layer1 = 2
layer2 = 64
layer3 = 1


#Create layers
dense1 = Layer_Dense(layer1, layer2, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
dense2 = Layer_Dense(layer2, layer3)

#Create activation functions
activation1 = Activation_ReLU()

activation2 = Activation_Sigmoid()

#Create loss function
loss_function = Loss_BinaryCrossentropy()

#Create optimizer
optimizer = Optimizer_Adam(decay=5e-7)

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

    activation2.forward(dense2.output)

    data_loss = loss_function.forward(activation2.output, y)

    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)

    loss = data_loss + regularization_loss

    #Calculate accuracy from output of activation2 and targets
    #Part in the brackets returns a binary mask - array consisting of True/False values, multiplying it by 1 changes it into array of 1s and 0s
    predictions = (activation2.output >= 0.5) * 1
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch} ' + 
              f'acc: {accuracy:.3f} ' + 
              f'loss: {loss:.3f} ' + 
              f'(data_loss: {data_loss:.3f} ' + 
              f'reg_loss: {regularization_loss:.3f}) '
              f'lr: {optimizer.current_learning_rate} ')
        
        all_loss.append(loss)
        all_accuracy.append(accuracy)
        all_epoch.append(epoch)

    #Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    #Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()




#Create graphs
graph_maker = Graph_maker()

#Example, guess, loss and accuracy graphs
graph_maker.graph_create_scatter(X[:,0], X[:,1], c=y, graph_name="Gerçek veri")
graph_maker.graph_create_scatter(X[:,0], X[:,1], c=(activation2.output >= 0.5).ravel(), graph_name="Tahmin")
graph_maker.graph_create_plot(all_epoch, all_loss, graph_name="Loss graph")
graph_maker.graph_create_plot(all_epoch, all_accuracy, graph_name="Accuracy graph")



#Validate the model

#Create test data 
X_test, y_test = spiral_data(samples=1000, classes=2)

y_test = y_test.reshape(-1, 1)

#Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.forward(activation2.output, y_test)

#Calculate accuracy from output of activation2 and targets calculate values along first axis
predictions = (activation2.output >= 0.5) * 1
accuracy = np.mean(predictions==y)

print(f'validation, acc, {accuracy:.3}, loss: {loss:.3}')


#Test graphs
graph_maker.graph_create_scatter(X_test[:,0], X_test[:,1], c=y_test, graph_name="Test verisi")
graph_maker.graph_create_scatter(X_test[:,0], X_test[:,1], c=(activation2.output >= 0.5).ravel(), graph_name="Test verisi tahmin")

#Show graph and clear data
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