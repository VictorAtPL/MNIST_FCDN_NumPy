from typing import List

import cupy as np

from simple_framework.layers.layer import Layer


class SequentialLayer(Layer):

    def __init__(self, layers: List[Layer]) -> None:
        self.layers: List[Layer] = layers

    def forward(self, input_tensor: np.ndarray, is_training: bool = True) -> np.ndarray:
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor, is_training)

        return input_tensor

    def backward(self, next_layer_derivative: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            next_layer_derivative = layer.backward(next_layer_derivative)

        return next_layer_derivative
