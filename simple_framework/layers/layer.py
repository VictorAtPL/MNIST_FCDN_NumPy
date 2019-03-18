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
    def forward(self, input_tensor: np.ndarray, is_training: bool = True) -> np.ndarray:
        """

        :param input_tensor: array of shape (batch_size, input_shape)
        :param is_training: specifies whether forward pass is made during training or not
        :return: array of shape (batch_size, output_shape)
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, next_layer_derivative: np.ndarray) -> np.ndarray:
        raise NotImplementedError
