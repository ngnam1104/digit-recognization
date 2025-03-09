import numpy as np
from .base_layer import BaseLayer
class Flatten(BaseLayer):
    def forward(self, X):
        """Chuyển tensor nhiều chiều thành vector một chiều."""
        self.input_shape = X.shape  # Lưu lại shape để sử dụng khi backward
        return X.flatten()

    def backward(self, dY, *args):
        """Reshape gradient về lại kích thước ban đầu."""
        return dY.reshape(self.input_shape)
