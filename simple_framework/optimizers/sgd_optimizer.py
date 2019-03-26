import numpy as np

from simple_framework.layers.layer import Layer
from simple_framework.optimizers.optimizer import Optimizer


class SGDOptimizer(Optimizer):

    learning_rate: float

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def do_back_propagation(self, layer: Layer, golden_labels: np.ndarray) -> None:
        layer.backward(golden_labels)

    def apply_gradients(self, tensor: np.ndarray, gradients: np.ndarray) -> None:
        tensor -= self.learning_rate * gradients

