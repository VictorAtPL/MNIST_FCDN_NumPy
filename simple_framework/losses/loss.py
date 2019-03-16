from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):

    @abstractmethod
    def count(self, tensor: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        raise NotImplementedError
