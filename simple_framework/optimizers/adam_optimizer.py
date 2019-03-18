from typing import Dict

import numpy as np

from simple_framework.layers.layer import Layer
from simple_framework.optimizers.optimizer import Optimizer


class AdamOptimizer(Optimizer):
    """Inspired by https://raw.githubusercontent.com/sagarvegad/Adam-optimizer/master/Adam.py"""

    learning_rate: float

    beta_1: float

    beta_2: float

    epsilon: int = 1e-8

    iteration: int = 0

    m_ts: Dict[int, np.ndarray] = {}

    v_ts: Dict[int, np.ndarray] = {}

    def __init__(self, learning_rate: float = 0.01, beta_1: float = 0.9, beta_2: float = 0.999) -> None:
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def do_back_propagation(self, layer: Layer, golden_labels: np.ndarray) -> None:
        layer.backward(golden_labels)

    def apply_gradients(self, tensor: np.ndarray, gradients: np.ndarray) -> None:
        self.iteration += 1

        id_of_weights_tensor = id(tensor)

        if id_of_weights_tensor not in self.m_ts:
            self.m_ts[id_of_weights_tensor] = np.zeros(tensor.shape, dtype=np.float32)
            self.v_ts[id_of_weights_tensor] = np.zeros(tensor.shape, dtype=np.float32)

        self.m_ts[id_of_weights_tensor] = self.beta_1 * self.m_ts[id_of_weights_tensor] + (1 - self.beta_1) * gradients
        self.v_ts[id_of_weights_tensor] = self.beta_2 * self.v_ts[id_of_weights_tensor] + (1 - self.beta_2) \
                                          * np.square(gradients)

        m_ts_corrected = self.m_ts[id_of_weights_tensor] / (1 - np.power(self.beta_1, self.iteration))
        v_ts_corrected = self.v_ts[id_of_weights_tensor] / (1 - np.power(self.beta_2, self.iteration))

        tensor -= (self.learning_rate * m_ts_corrected) / (np.sqrt(v_ts_corrected) + self.epsilon)

