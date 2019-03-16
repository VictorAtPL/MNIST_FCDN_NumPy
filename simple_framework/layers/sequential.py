from typing import List

import numpy as np

from simple_framework.layers.layer import Layer
from simple_framework.optimizers.optimizer import Optimizer


class SequentialLayer(Layer):

    def __init__(self, layers: List[Layer]) -> None:
        self.layers: List[Layer] = layers

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return input_tensor

    def backward(self, next_layer_derivative: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            next_layer_derivative = layer.backward(next_layer_derivative)

        return next_layer_derivative