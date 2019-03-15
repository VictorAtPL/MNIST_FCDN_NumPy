import numpy as np

from simple_framework.layer import Layer


class SoftmaxLayer(Layer):
    def forward(self, tensor: np.ndarray) -> np.ndarray:
        exp_tensor = np.exp(tensor)
        return exp_tensor / np.sum(exp_tensor, axis=1, keepdims=True)

    def backward(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError
