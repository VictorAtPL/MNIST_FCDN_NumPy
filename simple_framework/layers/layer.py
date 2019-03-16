from abc import ABC
from typing import Optional

import numpy as np

from simple_framework.optimizers.optimizer import Optimizer


class Layer(ABC):

    forward_cache: Optional[np.ndarray] = None

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        self.forward_cache = tensor
        return tensor

    def backward(self, tensor: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        raise NotImplementedError
