import numpy as np
from .base_layer import BaseLayer
class MaxPooling(BaseLayer):
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride
    
    def forward(self, input):
        self.last_input = input
        output_h = input.shape[1] // self.size
        output_w = input.shape[2] // self.size
        output = np.zeros((input.shape[0], output_h, output_w))
        
        for f in range(input.shape[0]):
            for i in range(output_h):
                for j in range(output_w):
                    output[f, i, j] = np.max(input[f, i*self.size:(i+1)*self.size, j*self.size:(j+1)*self.size])
        return output
    
    def backward(self, dL_dout, *args):
        dL_dinput = np.zeros(self.last_input.shape)
        for f in range(self.last_input.shape[0]):
            for i in range(dL_dout.shape[1]):
                for j in range(dL_dout.shape[2]):
                    region = self.last_input[f, i*self.size:(i+1)*self.size, j*self.size:(j+1)*self.size]
                    max_val = np.max(region)
                    for m in range(self.size):
                        for n in range(self.size):
                            if region[m, n] == max_val:
                                dL_dinput[f, i*self.size + m, j*self.size + n] += dL_dout[f, i, j]
        return dL_dinput
