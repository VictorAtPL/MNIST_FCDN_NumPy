from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):

    @abstractmethod
    def minimize(self, avg_loss: np.float32):
        raise NotImplementedError
