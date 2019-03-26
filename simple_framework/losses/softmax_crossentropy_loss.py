import cupy as np

from simple_framework.losses.loss import Loss
from utils import get_logger


class SoftmaxCrossEntropyLoss(Loss):

    lmbda: float

    logger = get_logger()

    def __init__(self, lmbda: float) -> None:
        self.lmbda = lmbda

    def forward(self, input_tensor: np.ndarray, is_training: bool = True) -> np.ndarray:
        self.logger.debug("%s.%s", self.__class__.__name__, "forward")

        # - np.max gives numerical stability
        exp_tensor = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        exp_tensor_normalized = exp_tensor / np.sum(exp_tensor, axis=1, keepdims=True)

        self.do_cache(input_tensor, exp_tensor_normalized)
        return exp_tensor_normalized

    def backward(self, next_layer_derivative: np.ndarray) -> np.ndarray:
        self.logger.debug("%s.%s", self.__class__.__name__, "backward")

        return self.cache['output_tensor'] - next_layer_derivative

    def get_overall(self, golden_one_hot: np.ndarray) -> np.ndarray:
        batch_size = self.cache['output_tensor'].shape[0]

        stable_output_cache = np.max(self.cache['output_tensor'], np.int32(1e-15))

        all_weights = self.global_cache['weights']
        weights_norm = sum([np.sum(np.dot(tensor, np.transpose(tensor))) for tensor in all_weights])
        self.logger.debug("L2 of all weights matrix %f", weights_norm)

        return -1. / batch_size * np.sum(golden_one_hot * np.log(stable_output_cache)) \
               + self.lmbda / (2 * batch_size) * weights_norm
