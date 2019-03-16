import numpy as np

from simple_framework.layers.layer import Layer
from utils import get_logger


class StandardizationLayer(Layer):

    logger = get_logger()

    def __init__(self, mean: np.float32, variance: np.float32) -> None:
        self.mean: np.float32 = mean
        self.variance: np.float32 = variance

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.logger.debug("%s.%s", self.__class__.__name__, "forward")
        standardized_tensor = (input_tensor - self.mean) / self.variance

        self.do_cache(input_tensor, standardized_tensor)
        return standardized_tensor

    def backward(self, next_layer_derivative: np.ndarray) -> np.ndarray:
        self.logger.debug("%s.%s", self.__class__.__name__, "backward")

        return next_layer_derivative