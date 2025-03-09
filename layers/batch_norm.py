import numpy as np
from .base_layer import BaseLayer
class BatchNormalization(BaseLayer):
    def __init__(self, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None

    def forward(self, X, training=True):
        if self.gamma is None:
            self.gamma = np.ones((1, X.shape[1]))
            self.beta = np.zeros((1, X.shape[1]))
            self.running_mean = np.zeros((1, X.shape[1]))
            self.running_var = np.ones((1, X.shape[1]))

        if training:
            mean = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)
            X_norm = (X - mean) / np.sqrt(var + self.epsilon)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        return self.gamma * X_norm + self.beta

    def backward(self, dY, X):
        N, D = X.shape
        mean = np.mean(X, axis=0, keepdims=True)
        var = np.var(X, axis=0, keepdims=True)

        X_norm = (X - mean) / np.sqrt(var + self.epsilon)
        dX_norm = dY * self.gamma

        dvar = np.sum(dX_norm * (X - mean) * -0.5 * (var + self.epsilon) ** (-3/2), axis=0, keepdims=True)
        dmean = np.sum(dX_norm * -1 / np.sqrt(var + self.epsilon), axis=0, keepdims=True) + dvar * np.sum(-2 * (X - mean), axis=0, keepdims=True) / N

        dX = dX_norm / np.sqrt(var + self.epsilon) + dvar * 2 * (X - mean) / N + dmean / N
        dgamma = np.sum(dY * X_norm, axis=0, keepdims=True)
        dbeta = np.sum(dY, axis=0, keepdims=True)

        return dX, dgamma, dbeta
