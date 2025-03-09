import numpy as np

def cross_entropy_loss(output, y):
    return -np.sum(y * np.log(output + 1e-10))  # Tránh log(0)