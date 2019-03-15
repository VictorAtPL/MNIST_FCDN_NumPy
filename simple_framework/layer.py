import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError
