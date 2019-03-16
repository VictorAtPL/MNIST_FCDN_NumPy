from typing import Optional

import numpy as np

from dataset import Dataset
from dataset_type import DatasetType
from simple_framework.layers.dense_layer import DenseLayer
from simple_framework.layers.layer import Layer
from simple_framework.layers.relu_layer import ReluLayer
from simple_framework.layers.sequential import SequentialLayer
from simple_framework.layers.standardization_layer import StandardizationLayer
from simple_framework.losses.softmax_crossentropy_loss import SoftmaxCrossEntropyLoss
from simple_framework.losses.loss import Loss
from simple_framework.optimizers.optimizer import Optimizer
from simple_framework.optimizers.sgd_optimizer import SGDOptimizer
from simple_framework.utils import convert_to_one_hot
from utils import get_logger


class Model:

    logger = get_logger()

    def __init__(self, batch_size: int, dataset: Dataset, weight_decay_lambda: float, learning_rate: float) -> None:
        super().__init__()

        self.batch_size: int = batch_size
        self.dataset: Dataset = dataset

        self.loss: Loss = SoftmaxCrossEntropyLoss(weight_decay_lambda)

        self.optimizer: Optimizer = SGDOptimizer(learning_rate=learning_rate)

        self.architecture: Layer = SequentialLayer([
            StandardizationLayer(np.float32(127), np.float32(255)),
            DenseLayer(128, weight_decay_lambda, optimizer=self.optimizer),
            ReluLayer(),
            DenseLayer(10, weight_decay_lambda, optimizer=self.optimizer),
            self.loss
        ])

    def train(self, epochs: int = 1, with_validation: bool = False):

        for epoch in range(1, epochs + 1):

            epoch_loss_list = []
            for batch_no, (features, golden_labels) in enumerate(self.dataset.get_batch(DatasetType.TRAINING,
                                                                                        batch_size=self.batch_size)):
                probabilities = self.architecture.forward(features)

                golden_one_hot = convert_to_one_hot(golden_labels, class_number=probabilities.shape[1])
                avg_loss = self.loss.get_overall(golden_one_hot)
                epoch_loss_list.append(avg_loss)

                self.optimizer.do_back_propagation(self.architecture, golden_one_hot)

            self.logger.info("epoch %d - training loss %f", epoch, np.mean(epoch_loss_list))

            if with_validation:
                self.validate(epoch)

    def validate(self, epoch: Optional[int] = None):
        epoch_loss_list = []
        for features, golden_labels in self.dataset.get_batch(DatasetType.VALIDATION):

            probabilities = self.architecture.forward(features)

            golden_one_hot = convert_to_one_hot(golden_labels, class_number=probabilities.shape[1])
            avg_loss = self.loss.get_overall(golden_one_hot)
            epoch_loss_list.append(avg_loss)

        if epoch:
            self.logger.info("epoch %d - validation loss %f ", epoch, np.mean(epoch_loss_list))
        else:
            self.logger.info("validation loss %f ", np.mean(epoch_loss_list))
