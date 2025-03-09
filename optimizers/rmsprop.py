import numpy as np

class RMSProp:
    def __init__(self, lr=0.01, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.s = {}  # Lưu trữ giá trị động lượng của gradient

    def update(self, params, grads):
        for i in range(len(params)):
            self.s[i] = self.beta * self.s.get(i, 0) + (1 - self.beta) * (grads[i] ** 2)
            params[i] -= self.lr * grads[i] / (np.sqrt(self.s[i]) + self.epsilon)