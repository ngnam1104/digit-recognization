import numpy as np

def mse_loss(output, y):
    return np.mean((output - y) ** 2)