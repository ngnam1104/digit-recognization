import numpy as np
from optimizers.adam import Adam
from utils.param_init import param_initializer
from .base_layer import BaseLayer
class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size, weight_init="random"):
        self.weights = param_initializer((input_size, output_size), method=weight_init)
        self.biases = np.zeros(output_size)
        self.weight_init = weight_init

    def forward(self, input_flat):
        """Nhận đầu vào đã flatten sẵn"""
        self.last_input = input_flat  # Lưu lại đầu vào để sử dụng khi backward
        return np.dot(input_flat, self.weights) + self.biases

    def backward(self, dL_dout, optimizer):
        dL_dweights = np.outer(self.last_input, dL_dout) 
        dL_dbiases = dL_dout
        dL_dinput = np.dot(dL_dout, self.weights.T)  

        optimizer.update(self.weights, dL_dweights)
        optimizer.update(self.biases, dL_dbiases)
        return dL_dinput
