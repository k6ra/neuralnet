import numpy as np
from layers import *

class NeuralNet:
    def __init__(self, weight_init_std=0.01):
        self.layers = []
        self.affineLayers = []
        self.lastLayer = SoftmaxWithLoss()
        self.weight_init_std = weight_init_std

    def add_affine(self, input_size, output_size):
        w = np.random.randn(input_size, output_size) * self.weight_init_std
        b = np.zeros(output_size)
        layer = Affine(w, b)
        self.layers.append(layer)
        self.affineLayers.append(layer)

    def add_active(self, func_name):
        if func_name == "relu":
            self.layers.append(Relu())
        elif func_name == "sigmoid":
            self.layers.append(Sigmoid())

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / x.shape[0]

        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x, t)

        for layer in self.affineLayers:
            layer.dw = numerical_gradient(loss_W, layer.w)
            layer.db = numerical_gradient(loss_W, layer.b)

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers)
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

    def fit(self, x, t, learning_rate=0.1):
        self.gradient(x, t)

        for layer in self.affineLayers:
            layer.w -= learning_rate * layer.dw
            layer.b -= learning_rate * layer.db
