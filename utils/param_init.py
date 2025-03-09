import numpy as np

def param_initializer(shape, method="random"):
    if method == "random":
        return np.random.randn(*shape) * 0.05
    elif method == "xavier":
        limit = np.sqrt(1.0 / np.prod(shape[1:]))  # 1 / sqrt(D_in)
        return np.random.uniform(-limit, limit, shape)
    elif method == "he":
        stddev = np.sqrt(2.0 / np.prod(shape[1:]))  # sqrt(2 / D_in)
        return np.random.randn(*shape) * stddev
    else:
        raise ValueError("Unsupported weight initialization method")  # <-- Fix lỗi xuống dòng
