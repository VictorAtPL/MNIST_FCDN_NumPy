import numpy as np

from simple_framework.optimizers.optimizer import Optimizer


class SGDOptimizer(Optimizer):
    def minimize(self, avg_loss: np.float32):
        raise NotImplementedError
