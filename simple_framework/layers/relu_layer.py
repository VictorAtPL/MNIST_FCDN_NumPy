import cupy as np

from simple_framework.layers.layer import Layer
from utils import get_logger


class ReluLayer(Layer):

    logger = get_logger()

    def forward(self, input_tensor: np.ndarray, is_training: bool = True) -> np.ndarray:
        self.logger.debug("%s.%s", self.__class__.__name__, "forward")

        result = np.maximum(input_tensor, 0)

        self.do_cache(input_tensor, result)
        return result

    def backward(self, next_layer_derivative: np.ndarray) -> np.ndarray:
        self.logger.debug("%s.%s", self.__class__.__name__, "backward")
        result = np.multiply(next_layer_derivative, np.array(self.cache['output_tensor'] > 0, dtype=np.int32))
        return result
