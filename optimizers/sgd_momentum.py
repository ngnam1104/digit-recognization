class SGD_Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}  # Lưu trữ động lượng

    def update(self, params, grads):
        for i in range(len(params)):
            self.v[i] = self.momentum * self.v.get(i, 0) - self.lr * grads[i]
            params[i] += self.v[i]  # Cập nhật tham số dựa trên động lượng