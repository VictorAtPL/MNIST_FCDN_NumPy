import numpy as np

from simple_framework.layer import Layer


class StandarizationLayer(Layer):

    def __init__(self, mean: np.float32, variance: np.float32) -> None:
        self.mean: np.float32 = mean
        self.variance: np.float32 = variance

        super().__init__()

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        return (tensor - self.mean) / self.variance

    def backward(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError
