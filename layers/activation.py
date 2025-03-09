import numpy as np
from .base_layer import BaseLayer
class ReLU(BaseLayer):
    def forward(self, input):
        self.last_input = input
        return np.maximum(0, input)

    def backward(self, dL_dout, *args):
        dL_dinput = dL_dout * (self.last_input > 0)
        return dL_dinput

class Softmax(BaseLayer):
    def forward(self, input):
        exps = np.exp(input - np.max(input))
        return exps / np.sum(exps)
    
    def backward(self, dL_dout, *args):
        return dL_dout

class Sigmoid(BaseLayer):
    def forward(self, input):
        self.last_output = 1 / (1 + np.exp(-input))
        return self.last_output

    def backward(self, dL_dout, *args):
        return dL_dout * self.last_output * (1 - self.last_output)

class LeakyReLU(BaseLayer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, input):
        self.last_input = input
        return np.where(input > 0, input, self.alpha * input)

    def backward(self, dL_dout, *args):
        dL_dinput = dL_dout * np.where(self.last_input > 0, 1, self.alpha)
        return dL_dinput