import numpy as np

from dataset import Dataset
from dataset_type import DatasetType
from simple_framework.layers.dense_layer import DenseLayer
from simple_framework.layers.layer import Layer
from simple_framework.layers.sequential import SequentialLayer
from simple_framework.layers.softmax_layer import SoftmaxLayer
from simple_framework.layers.standardization_layer import StandardizationLayer
from simple_framework.losses.crossentropy_loss import CrossEntropyLoss
from simple_framework.optimizers.sgd_optimizer import SGDOptimizer
from simple_framework.utils import convert_to_one_hot


class Model:
    def __init__(self, batch_size: int, dataset: Dataset) -> None:
        super().__init__()

        self.batch_size: int = batch_size
        self.dataset: Dataset = dataset

        self.architecture: Layer = SequentialLayer([
            StandardizationLayer(np.float32(127), np.float32(255)),
            DenseLayer(10),
            SoftmaxLayer()
        ])

    def train(self, epochs: int = 1, with_validation: bool = False):
        optimizer = SGDOptimizer()

        for _ in range(epochs):
            # TODO: Change to TRAINING
            for features, golden_labels in self.dataset.get_batch(DatasetType.VALIDATION):
                output = self.architecture.forward(features)
                loss = CrossEntropyLoss()

                golden_labels = convert_to_one_hot(golden_labels, class_number=output.shape[1])
                avg_loss = loss.count(output, golden_labels)
                self.architecture.backward(golden_labels, optimizer)

                pass

            if with_validation:
                self.validate()

    def validate(self):
        raise NotImplementedError
