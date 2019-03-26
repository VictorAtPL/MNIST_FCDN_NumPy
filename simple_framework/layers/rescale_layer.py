import cupy as np

from simple_framework.layers.layer import Layer
from utils import get_logger


class RescaleLayer(Layer):

    logger = get_logger()

    def __init__(self, subtrahend: np.float32, dividend: np.float32) -> None:
        self.subtrahend: np.float32 = subtrahend
        self.dividend: np.float32 = dividend

    def forward(self, input_tensor: np.ndarray, is_training: bool = True) -> np.ndarray:
        self.logger.debug("%s.%s", self.__class__.__name__, "forward")
        rescaled_tensor = (input_tensor - self.subtrahend) / self.dividend

        self.do_cache(input_tensor, rescaled_tensor)
        return rescaled_tensor

    def backward(self, next_layer_derivative: np.ndarray) -> np.ndarray:
        self.logger.debug("%s.%s", self.__class__.__name__, "backward")

        return next_layer_derivative
