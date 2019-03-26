from abc import abstractmethod

import numpy as np

from simple_framework.layers.layer import Layer


class Loss(Layer):
    @abstractmethod
    def forward(self, input_tensor: np.ndarray, is_training: bool = True) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, next_layer_derivative: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_overall(self, golden_one_hot: np.ndarray) -> np.ndarray:
        raise NotImplementedError
