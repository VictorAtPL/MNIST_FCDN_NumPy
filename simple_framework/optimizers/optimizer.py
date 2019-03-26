from abc import ABC, abstractmethod

import numpy as np

from simple_framework.layers.layer import Layer


class Optimizer(ABC):

    @abstractmethod
    def do_back_propagation(self, layer: Layer, golden_labels: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_gradients(self, tensor: np.ndarray, gradients: np.ndarray) -> None:
        raise NotImplementedError
