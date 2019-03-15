import numpy as np

from dataset import Dataset
from dataset_type import DatasetType
from simple_framework.dense_layer import DenseLayer
from simple_framework.layer import Layer
from simple_framework.sequential import SequentialLayer
from simple_framework.softmax_layer import SoftmaxLayer
from simple_framework.standarization_layer import StandarizationLayer


class Model:
    def __init__(self, batch_size: int, dataset: Dataset) -> None:
        super().__init__()

        self.batch_size: int = batch_size
        self.dataset: Dataset = dataset

        self.architecture: Layer = SequentialLayer([
            StandarizationLayer(np.float32(127), np.float32(255)),
            DenseLayer(10),
            SoftmaxLayer()
        ])

    def train(self, epochs: int = 1, with_validation: bool = False):
        for _ in range(epochs):
            for features, labels in self.dataset.get_batch(DatasetType.TRAINING):
                output = self.architecture.forward(features)
                pass

            if with_validation:
                self.validate()

    def validate(self):
        raise NotImplementedError
