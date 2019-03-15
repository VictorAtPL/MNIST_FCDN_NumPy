from typing import List

import numpy as np

from simple_framework.layer import Layer


class SequentialLayer(Layer):

    def __init__(self, layers: List[Layer]) -> None:
        self.layers: List[Layer] = layers

        super().__init__()

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            tensor = layer.forward(tensor)

        return tensor

    def backward(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError
