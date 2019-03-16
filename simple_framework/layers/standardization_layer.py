import numpy as np

from simple_framework.layers.layer import Layer
from simple_framework.optimizers.optimizer import Optimizer


class StandardizationLayer(Layer):

    def __init__(self, mean: np.float32, variance: np.float32) -> None:
        self.mean: np.float32 = mean
        self.variance: np.float32 = variance

        super().__init__()

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        return (tensor - self.mean) / self.variance

    def backward(self, tensor: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        raise NotImplementedError
