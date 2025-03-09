import numpy as np
from utils.param_init import param_initializer
from optimizers.adam import Adam
from .base_layer import BaseLayer
class Convolution(BaseLayer):
    def __init__(self, num_filters, kernel_size, stride=1, padding=0, weight_init="random"):
        self.num_filters = num_filters
        if isinstance(kernel_size, tuple):
            self.kernel_height, self.kernel_width = kernel_size
        else:
            self.kernel_height = self.kernel_width = kernel_size 
        self.stride = stride
        self.padding = padding
        self.weight_init = weight_init
        self.filters = param_initializer((num_filters, self.kernel_height, self.kernel_width), method=self.weight_init)
        self.biases = np.zeros(num_filters)
        
    
    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(0, h - self.kernel_height + 1, self.stride):
            for j in range(0, w - self.kernel_width + 1, self.stride):
                region = image[i:(i + self.kernel_height), j:(j + self.kernel_width)]
                yield i, j, region

    def forward(self, input):
        self.last_input = input
        if self.padding > 0:
            input = np.pad(input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        h, w = input.shape[1:] 
        output_dim = (h - self.kernel_height) // self.stride + 1
        output = np.zeros((self.num_filters, output_dim, output_dim))

        for f in range(self.num_filters):
            for i, j, region in self.iterate_regions(input[0]):
                output[f, i, j] = np.sum(region * self.filters[f]) + self.biases[f]
        return output
    
    def backward(self, dL_dout, optimizer):
        dL_dfilters = np.zeros(self.filters.shape)
        dL_dbiases = np.zeros(self.biases.shape)
        dL_dinput = np.zeros(self.last_input.shape)

        for f in range(self.num_filters):
            for i, j, region in self.iterate_regions(self.last_input[0]):
                dL_dfilters[f] += dL_dout[f, i, j] * region
                dL_dbiases[f] += dL_dout[f, i, j]
                dL_dinput[0, i:i+self.kernel_height, j:j+self.kernel_width] += dL_dout[f, i, j] * self.filters[f]
        optimizer.update(self.filters, dL_dfilters)
        optimizer.update(self.biases, dL_dbiases)
        return dL_dinput