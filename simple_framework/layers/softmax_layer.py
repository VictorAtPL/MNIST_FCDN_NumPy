import numpy as np

from simple_framework.layers.layer import Layer
from simple_framework.optimizers.optimizer import Optimizer


class SoftmaxLayer(Layer):
    def forward(self, tensor: np.ndarray) -> np.ndarray:
        # - np.max gives numerical stability
        exp_tensor = np.exp(tensor - np.max(tensor, axis=1))
        exp_tensor_normalized = exp_tensor / np.sum(exp_tensor, axis=1, keepdims=True)
        return super(SoftmaxLayer, self).forward(exp_tensor_normalized)

    def backward(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError
