import numpy as np

from simple_framework.losses.loss import Loss


class CrossEntropyLoss(Loss):
    def count(self, tensor: np.ndarray, ground_truth: np.ndarray):
        return - np.average(np.sum(ground_truth * np.log(tensor) +
                                   (1 - ground_truth) * np.log(1 - tensor), axis=1))
