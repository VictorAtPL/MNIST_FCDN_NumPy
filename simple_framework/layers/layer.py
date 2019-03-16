from abc import ABC, abstractmethod
from typing import Optional, Dict, List

import numpy as np


class Layer(ABC):

    cache: Dict[str, Optional[np.ndarray]]

    global_cache: Dict[str, List[np.ndarray]] = {
        "weights": []
    }

    def do_cache(self, input_tensor: np.ndarray, output_tensor: np.ndarray) -> None:
        self.cache: Dict[str, Optional[np.ndarray]] = {
            "input_tensor": input_tensor,
            "output_tensor": output_tensor
        }

    @abstractmethod
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, next_layer_derivative: np.ndarray) -> np.ndarray:
        raise NotImplementedError
