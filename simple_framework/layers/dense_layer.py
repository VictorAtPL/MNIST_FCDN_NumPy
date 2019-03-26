from typing import Optional

import cupy as np

from simple_framework.layers.layer import Layer
from simple_framework.optimizers.optimizer import Optimizer
from utils import get_logger


class DenseLayer(Layer):

    logger = get_logger()

    # input features (neurons) number
    input_number: int = None

    # output features (neurons) number
    output_number: int = None

    # lmbda parameter for weight decay
    lmbda: float = 0.0

    # optimizer to be used while applying gradients
    optimizer: Optional[Optimizer] = None

    # array of shape (input_shape, output_shape)
    weights: Optional[np.ndarray] = None

    # bias
    bias: Optional[np.ndarray] = None

    def __init__(self, output_number: int, lmbda: float, optimizer: Optional[Optimizer] = None) -> None:
        self.output_number = output_number
        self.lmbda = lmbda
        self.optimizer = optimizer

    def forward(self, input_tensor: np.ndarray, is_training: bool = True) -> np.ndarray:
        self.logger.debug("%s.%s", self.__class__.__name__, "forward")

        if not self.input_number:
            self.input_number: int = input_tensor.shape[1]

            self.weights = np.random.randn(self.input_number, self.output_number) * 0.01
            self.global_cache['weights'].append(self.weights)
            self.bias = np.zeros((1, self.output_number))

        assert input_tensor.shape[1] == self.weights.shape[0]

        result = np.dot(input_tensor, self.weights) + self.bias

        self.do_cache(input_tensor, result)
        return result

    def backward(self, next_layer_derivative: np.ndarray) -> np.ndarray:
        """

        :param next_layer_derivative: array of shape (batch_size, output_shape)
        :return: array of shape (batch_size, input_shape)
        """
        self.logger.debug("%s.%s", self.__class__.__name__, "backward")

        batch_size = self.cache['input_tensor'].shape[0]
        # array of shape (input_shape, output_shape) = (batch_size, input_shape).T @ (batch_size, output_shape)
        weight_decay_part = (self.lmbda / batch_size) * self.weights
        weights_derivative: np.ndarray = (1. / batch_size) \
                                         * np.dot(np.transpose(self.cache['input_tensor']), next_layer_derivative) \
                                         + weight_decay_part

        # array of shape (1, output_shape)
        bias_derivative: np.ndarray = (1. / batch_size) \
                                      * np.sum(next_layer_derivative, axis=0, keepdims=True)

        self.optimizer.apply_gradients(self.weights, weights_derivative)
        self.optimizer.apply_gradients(self.bias, bias_derivative)

        # array of shape (batch_size, input_shape) = (batch_size, output_shape) @ (input_shape, output_shape).T
        return np.dot(next_layer_derivative, np.transpose(self.weights))
