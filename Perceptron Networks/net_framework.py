import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

#General Perceptron Neural Network Framework
#With sigmoid activation function and randomly 
#initialized weights

#Network Implementation

class Neural_Network(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize, learning_rate, probabability = False):
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
        binary : float
        	Whether the output is binary or continuous
        '''
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.probabability = probabability
        #Defining layers
        self.layers = []
        prev_size = inputSize
        for hidden_num in hiddenSize:
        	self.layers.append(nn.Linear(prev_size, hidden_num))
        	self.layers.append(nn.ELU())
        	prev_size = hidden_num
        self.layers.append(nn.Linear(prev_size, outputSize))
        self.layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*self.layers)
        #Using Binary cross entropy loss to train
        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, X, binary = False):
        y = self.model(X)
        if self.probabability:
        	y = y/torch.sum(y)
        if binary:
        	y = torch.round(y)
        return y

    def l1error(self, y_true, y_pred):
    	#Absolute Error
    	return torch.mean(torch.abs(y_true - y_pred)).item()

    def l2error(self, y_true, y_pred):
    	#Squared error
    	return torch.mean((y_true - y_pred)**2).item()

    def train(self, X, y):
        # Training using binary cross entropy loss (one iteration of training)
        output = self.forward(X)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()  # Does the update
    
    def saveWeights(self, model, path):
        # we will use the PyTorch internal storage functions
        torch.save(model.state_dict(), path)
        
    def predict(self, inp):
        return self.forward(inp)


