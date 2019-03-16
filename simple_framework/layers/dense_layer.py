from typing import Optional

import numpy as np

from simple_framework.layers.layer import Layer
from simple_framework.optimizers.optimizer import Optimizer


class DenseLayer(Layer):
    input_shape: int = None

    # array of shape (input_shape, output_shape)
    weights: Optional[np.ndarray] = None

    def __init__(self, output_shape: int) -> None:
        self.output_shape: int = output_shape

        super().__init__()

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        """

        :param tensor: array of shape (batch_size, input_shape)
        :return: array of shape (batch_size, output_shape)
        """

        if not self.input_shape:
            self.input_shape: int = tensor.shape[1]

            self.weights = np.random.randn(self.input_shape, self.output_shape) * 0.001

        assert tensor.shape[1] == self.weights.shape[0]

        return np.dot(tensor, self.weights)

    def backward(self, tensor: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        raise NotImplementedError
