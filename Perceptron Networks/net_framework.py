import torch
import torch.nn as nn
import numpy as np

#General Perceptron Neural Network Framework
#With sigmoid activation function and randomly 
#initialized weights

#Network Implementation

class Neural_Network(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize, learning_rate):
        super(Neural_Network, self).__init__()
        '''
        Params
        ------
        inputSize : int
            Size of input layer
        outputSize : int
            Size of output layer
        hiddenSize : 1D list of ints
            Size of each hidden layer
        learning_rate : float
            Learning rate
        '''
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.learning_rate = learning_rate
        # weights
        self.weights = [torch.randn(self.inputSize, self.hiddenSize[0]).float()]
        self.biases = []
        for i in range(len(self.hiddenSize)):
            if i < len(self.hiddenSize) - 1:
                self.weights.append(torch.randn(self.hiddenSize[i], self.hiddenSize[i + 1]).float())
            elif i == len(self.hiddenSize) - 1:
                self.weights.append(torch.randn(self.hiddenSize[i], self.outputSize).float())
            self.biases.append(torch.randn(self.hiddenSize[i], 1).float().T)
        self.biases.append(torch.randn(self.outputSize, 1).float().T)  
        
        
    def forward(self, X):
        self.z_arr = [X]
        self.z = X
        for i in range(len(self.weights)):
            self.z = torch.matmul(self.z, self.weights[i])
            self.z = self.z + self.biases[i] 
            self.z = self.sigmoid(self.z) # activation function
            self.z_arr.append(self.z)
        return self.z
        
    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))
    
    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)
    
    def backward(self, X, y):
        self.error = y - self.z_arr[-1] # error in output
        self.delta = self.error * self.sigmoidPrime(self.z_arr[-1]) # derivative of sig to error 
        for i in range(len(self.weights)):  
            self.weights[-i-1] = self.weights[-i-1] + torch.matmul(torch.t(self.z_arr[-i-2]), self.delta) * self.learning_rate
            self.biases[-i-1] = self.biases[-i-1] + torch.matmul(torch.from_numpy(np.ones(self.z_arr[-i-2].size()[0])).float().T, self.delta)* self.learning_rate
            self.error = torch.matmul(self.delta, torch.t(self.weights[-i-1]))
            self.delta = self.error * self.sigmoidPrime(self.z_arr[-i-2])
        
        
    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y)
        
    def saveWeights(self, model, path):
        # we will use the PyTorch internal storage functions
        torch.save(model, path)
        
    def predict(self, inp):
        return self.forward(inp)


