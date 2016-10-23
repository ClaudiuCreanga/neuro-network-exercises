import numpy as np
import matplotlib.pyplot as plt
class NeuralNetwork(object):

    def __init__(self):

        #  define hyperparams
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):

        #  propagate inputs through the network
        self.z2 = np.dot(X, self.W1)  # z^2 = XW^1  second layer
        self.a2 = self.sigmoid(self.z2)  # a^2 = f(z^2)
        self.z3 = np.dot(self.a2, self.W2)  # z^3 = a^2W^2  third layer
        yHat = self.sigmoid(self.z3)  # y = f(z^3)
        return yHat

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))

myNeuro = NeuralNetwork()
print(myNeuro.forward((3,5)))
