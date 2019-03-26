import numpy as np

from simple_framework.layers.layer import Layer
from utils import get_logger


class DropoutLayer(Layer):

    logger = get_logger()

    keep_probability: float

    mask: np.ndarray

    def __init__(self, keep_probability: float = 1.0) -> None:
        self.keep_probability = keep_probability

    def forward(self, input_tensor: np.ndarray, is_training: bool = True) -> np.ndarray:
        self.logger.debug("%s.%s", self.__class__.__name__, "forward")

        if not is_training:
            self.mask = np.ones(input_tensor.shape)
        else:
            self.mask = np.random.rand(input_tensor.shape[0], input_tensor.shape[1]) < self.keep_probability

        result = input_tensor * self.mask

        self.do_cache(input_tensor, result)
        return result

    def backward(self, next_layer_derivative: np.ndarray) -> np.ndarray:
        self.logger.debug("%s.%s", self.__class__.__name__, "backward")

        result = next_layer_derivative * self.mask / self.keep_probability
        return result
