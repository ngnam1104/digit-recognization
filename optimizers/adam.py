import numpy as np
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment vector
        self.v = {}  # Second moment vector

    def update(self, params, grads):
        if id(params) not in self.m:
            self.m[id(params)] = np.zeros_like(params)
            self.v[id(params)] = np.zeros_like(params)

        # Cập nhật moment
        self.m[id(params)] = self.beta1 * self.m[id(params)] + (1 - self.beta1) * grads
        self.v[id(params)] = self.beta2 * self.v[id(params)] + (1 - self.beta2) * (grads ** 2)

        # Điều chỉnh bias
        m_hat = self.m[id(params)] / (1 - self.beta1)
        v_hat = self.v[id(params)] / (1 - self.beta2)

        # Cập nhật tham số
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
