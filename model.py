from typing import Optional, Tuple

import numpy as np

from dataset import Dataset
from dataset_type import DatasetType
from plot_drawer import PlotDrawer
from simple_framework.layers.dense_layer import DenseLayer
from simple_framework.layers.layer import Layer
from simple_framework.layers.relu_layer import ReluLayer
from simple_framework.layers.sequential import SequentialLayer
from simple_framework.layers.standardization_layer import StandardizationLayer
from simple_framework.losses.softmax_crossentropy_loss import SoftmaxCrossEntropyLoss
from simple_framework.losses.loss import Loss
from simple_framework.optimizers.adam_optimizer import AdamOptimizer
from simple_framework.optimizers.optimizer import Optimizer
from simple_framework.utils import convert_to_one_hot
from utils import get_logger


class Model:

    logger = get_logger()

    def __init__(self, batch_size: int, dataset: Dataset, weight_decay_lambda: float, learning_rate: float) -> None:
        super().__init__()

        self.batch_size: int = batch_size
        self.dataset: Dataset = dataset

        self.loss: Loss = SoftmaxCrossEntropyLoss(weight_decay_lambda)

        self.optimizer: Optimizer = AdamOptimizer(learning_rate=learning_rate)

        self.architecture: Layer = SequentialLayer([
            StandardizationLayer(np.float32(127), np.float32(255)),
            DenseLayer(256, weight_decay_lambda, optimizer=self.optimizer),
            ReluLayer(),
            DenseLayer(128, weight_decay_lambda, optimizer=self.optimizer),
            ReluLayer(),
            DenseLayer(64, weight_decay_lambda, optimizer=self.optimizer),
            ReluLayer(),
            DenseLayer(32, weight_decay_lambda, optimizer=self.optimizer),
            ReluLayer(),
            DenseLayer(10, weight_decay_lambda, optimizer=self.optimizer),
            self.loss
        ])

    def train(self, epochs: int = 1, with_validation: bool = False, plot_graphs: bool = False):
        plot_drawer = PlotDrawer(plot_graphs, ["Training loss", "Training accuracy",
                                               "Validation loss", "Validation accuracy"])

        for epoch in range(1, epochs + 1):

            epoch_training_loss, epoch_training_accuracy = self._do_training_epoch()
            self.logger.info("epoch %d - training loss %f - training accuracy %f", epoch, epoch_training_loss,
                             epoch_training_accuracy)

            if with_validation:
                epoch_validation_loss, epoch_validation_accuracy = self._do_validation_epoch()
                self.logger.info("epoch %d - validation loss %f - validation accuracy %f", epoch, epoch_validation_loss,
                                 epoch_validation_accuracy)

                plot_drawer.add_points(epoch, (epoch_training_loss, epoch_training_accuracy,
                                               epoch_validation_loss, epoch_validation_accuracy))
            else:
                plot_drawer.add_points(epoch, (epoch_training_loss, epoch_training_accuracy))

            plot_drawer.plot()

    def _do_training_epoch(self) -> Tuple[float, float]:
        epoch_loss_list = []
        epoch_well_predicted = 0
        epoch_all_examples = 0
        for features, golden_labels in self.dataset.get_batch(DatasetType.TRAINING, batch_size=self.batch_size):
            probabilities = self.architecture.forward(features)

            predictions = np.argmax(probabilities, axis=1)
            golden_classes = np.squeeze(golden_labels, axis=1)

            well_predicted = predictions == golden_classes
            epoch_well_predicted += int(np.sum(well_predicted))
            epoch_all_examples += well_predicted.shape[0]

            golden_one_hot = convert_to_one_hot(golden_labels, class_number=probabilities.shape[1])
            avg_loss = self.loss.get_overall(golden_one_hot)
            epoch_loss_list.append(avg_loss)

            self.optimizer.do_back_propagation(self.architecture, golden_one_hot)

        epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
        epoch_accuracy = epoch_well_predicted / epoch_all_examples

        return epoch_loss, epoch_accuracy

    def _do_validation_epoch(self) -> Tuple[float, float]:
        epoch_loss_list = []
        epoch_well_predicted = 0
        epoch_all_examples = 0
        for features, golden_labels in self.dataset.get_batch(DatasetType.VALIDATION):

            probabilities = self.architecture.forward(features)

            predictions = np.argmax(probabilities, axis=1)
            golden_classes = np.squeeze(golden_labels, axis=1)

            well_predicted = predictions == golden_classes
            epoch_well_predicted += int(np.sum(well_predicted))
            epoch_all_examples += well_predicted.shape[0]

            golden_one_hot = convert_to_one_hot(golden_labels, class_number=probabilities.shape[1])
            avg_loss = self.loss.get_overall(golden_one_hot)
            epoch_loss_list.append(avg_loss)

        epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
        epoch_accuracy = epoch_well_predicted / epoch_all_examples

        return epoch_loss, epoch_accuracy


