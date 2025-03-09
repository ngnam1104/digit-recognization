import numpy as np

def minmax_normalize(images):
    return images / 255.0

def augment(images):
    # Đảo ngược ảnh, xoay, dịch chuyển
    flipped = np.flip(images, axis=2)  
    return np.concatenate((images, flipped))

def std_normalize(images):
    mean = np.mean(images, axis=(0, 1, 2), keepdims=True)
    std = np.std(images, axis=(0, 1, 2), keepdims=True)
    return (images - mean) / (std + 1e-8)  # Thêm epsilon để tránh chia cho 0