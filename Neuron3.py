import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data

nnfs.init()

import matplotlib.pyplot as plt

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


#ReLU activation
class Activation_ReLU:
    #Forward pass
    def forward(self, inputs):
        #Calculate output values from inputs
        self.output = np.maximum(0, inputs)


#Softmax activation
class Activation_Softmax:
    #Forward pass
    def forward(self, inputs):
        #Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        #Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


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


#Create dataset
#X, y = spiral_data(samples=100, classes=3)
X, y = vertical_data(samples=100, classes=3)

#Neuron numbers
layer1 = 2
layer2 = 5
layer3 = 5
layer4 = 5
layer5 = 3

#Create layers
dense1 = Layer_Dense(layer1,layer2)
dense2 = Layer_Dense(layer2,layer3)
dense3 = Layer_Dense(layer3,layer4)
dense4 = Layer_Dense(layer4,layer5)

#Create activation functions
activation1 = Activation_ReLU()
activation2 = Activation_ReLU()

activation3 = Activation_Softmax()

#Create loss function
loss_function = Loss_CategoricalCrossentropy()

#Helper variables
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
best_dense3_weights = dense3.weights.copy()
best_dense3_biases = dense3.biases.copy()
best_dense4_weights = dense4.weights.copy()
best_dense4_biases = dense4.biases.copy()

all_losses = []
all_accuracys = []
all_iterations = []

for iteration in range(10000):
    #Generate a new set of weights for iteration
    dense1.weights += 0.05 * np.random.randn(layer1,layer2)
    dense1.biases += 0.05 * np.random.randn(1,layer2)
    dense2.weights += 0.05 * np.random.randn(layer2,layer3)
    dense2.biases += 0.05 * np.random.randn(1,layer3)
    dense3.weights += 0.05 * np.random.randn(layer3,layer4)
    dense3.biases += 0.05 * np.random.randn(1,layer4)
    dense4.weights += 0.05 * np.random.randn(layer4,layer5)
    dense4.biases += 0.05 * np.random.randn(1,layer5)

    #Perdorm a forward pass of the training data through this layer
    dense1.forward(X)
    dense2.forward(dense1.output)
    activation1.forward(dense2.output)
    dense3.forward(activation1.output)
    activation2.forward(dense3.output)
    dense4.forward(activation2.output)
    activation3.forward(dense4.output)

    #Perdorm a forward pass through activation function it takes the output of fourth layer here and returns loss
    loss =  loss_function.calculate(activation3.output, y)

    #Calculate accuracy from output of activation3 and targets calculate values along first axis
    predictions = np.argmax(activation3.output, axis=1)
    accuracy = np.mean(predictions==y)

    #If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print("Congratulations new lowest loss is found!!"," Iteration: ", iteration, "Loss: ", loss, "Acc: ", accuracy)

        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        best_dense3_weights = dense3.weights.copy()
        best_dense3_biases = dense3.biases.copy()
        best_dense4_weights = dense4.weights.copy()
        best_dense4_biases = dense4.biases.copy()
        lowest_loss = loss

        all_losses.append(loss)
        all_accuracys.append(accuracy)
        all_iterations.append(iteration)
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
        dense3.weights = best_dense3_weights.copy()
        dense3.biases = best_dense3_biases.copy()
        dense4.weights = best_dense4_weights.copy()
        dense4.biases = best_dense4_biases.copy()



#Calculate
dense1.forward(X)

dense2.forward(dense1.output)
activation1.forward(dense2.output)

dense3.forward(activation1.output)
activation2.forward(dense3.output)

dense4.forward(activation2.output)
activation3.forward(dense4.output)

print(activation3.output[:5])

loss = loss_function.calculate(activation3.output, y)

print(loss)

#Calculate accuracy from output of activation3 and targets calculate values along first axis
predictions = np.argmax(activation3.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
print(accuracy)




#Graph of the data
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, s=15, cmap='brg')
plt.title("Data")
plt.figure()
plt.scatter(X[:,0], X[:,1], c=np.argmax(activation3.output, axis=1), s=15, cmap='brg')
plt.title("Tahmin")

#Graph of the loss and accuracy
plt.figure()
plt.plot(all_iterations, all_losses)
plt.title("Loss")
plt.figure()
plt.plot(all_iterations, all_accuracys)
plt.title("Accuracy")

plt.show()