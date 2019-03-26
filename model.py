import time
from typing import Tuple, List, Dict, Any

import cupy as np

from dataset import Dataset
from dataset_type import DatasetType
from plot_drawer import PlotDrawer
from simple_framework.layers.dense_layer import DenseLayer
from simple_framework.layers.dropout_layer import DropoutLayer
from simple_framework.layers.layer import Layer
from simple_framework.layers.relu_layer import ReluLayer
from simple_framework.layers.rescale_layer import RescaleLayer
from simple_framework.layers.sequential_layer import SequentialLayer
from simple_framework.losses.loss import Loss
from simple_framework.losses.softmax_crossentropy_loss import SoftmaxCrossEntropyLoss
from simple_framework.optimizers.adam_optimizer import AdamOptimizer
from simple_framework.optimizers.optimizer import Optimizer
from simple_framework.helpers import convert_to_one_hot
from utils import get_logger


class Model:

    logger = get_logger()

    def __init__(self, dataset: Dataset, batch_size: int, neurons_in_hidden_blocks: List[int],
                 weight_decay_lambda: float, learning_rate: float, dropout_keep_prob: float) -> None:
        super().__init__()

        self.dataset: Dataset = dataset

        self.batch_size: int = batch_size

        self.loss: Loss = SoftmaxCrossEntropyLoss(weight_decay_lambda)

        self.optimizer: Optimizer = AdamOptimizer(learning_rate=learning_rate)
        # self.optimizer: Optimizer = SGDOptimizer(learning_rate=learning_rate)

        blocks = [
            RescaleLayer(np.float32(128), np.float32(128))
        ]

        for neurons_in_hidden_block in neurons_in_hidden_blocks:
            blocks.extend([
                DenseLayer(neurons_in_hidden_block, weight_decay_lambda, optimizer=self.optimizer),
                ReluLayer(),
                DropoutLayer(keep_probability=dropout_keep_prob),
            ])

        blocks.extend([
            DenseLayer(10, weight_decay_lambda, optimizer=self.optimizer),
            self.loss
        ])

        self.architecture: Layer = SequentialLayer(blocks)

    def train(self, epochs: int = 1, with_validation: bool = False, plot_graphs: bool = False):
        plot_drawer = PlotDrawer(plot_graphs, ["Training loss", "Training accuracy",
                                               "Validation loss", "Validation accuracy"])
        bad_predicted_list = {i: [] for i in range(10)}
        golden_classes_all = np.array([])
        probabilities_all = np.array([])

        for epoch in range(1, epochs + 1):
            start = time.time()

            epoch_training_loss, epoch_training_accuracy = self._do_training_epoch()
            self.logger.info("epoch %d - training loss %f - training accuracy %f", epoch, epoch_training_loss,
                             epoch_training_accuracy)

            if with_validation:
                epoch_validation_loss, epoch_validation_accuracy, bad_predicted_list, \
                 golden_classes_all, probabilities_all = self._do_validation_epoch()

                self.logger.info("epoch %d - validation loss %f - validation accuracy %f", epoch, epoch_validation_loss,
                                 epoch_validation_accuracy)

                plot_drawer.add_points(epoch, (epoch_training_loss, epoch_training_accuracy,
                                               epoch_validation_loss, epoch_validation_accuracy))
            else:
                plot_drawer.add_points(epoch, (epoch_training_loss, epoch_training_accuracy))

            end = time.time()
            self.logger.info("epoch %d - time taken %.2f", epoch, end - start)

            plot_drawer.plot()

        if with_validation:
            plot_drawer.plot_wrong_classified(bad_predicted_list)
            plot_drawer.plot_rocs(golden_classes_all, probabilities_all)

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

    def _do_validation_epoch(self) -> Tuple[float, float, Dict[Any, Any], List[Any], List[Any]]:
        bad_predicted_list = {i: [] for i in range(10)}

        probabilities_all = np.empty((0, 10))
        golden_classes_all = np.empty((0,))

        epoch_loss_list = []
        epoch_well_predicted = 0
        epoch_all_examples = 0
        for features, golden_labels in self.dataset.get_batch(DatasetType.VALIDATION):

            probabilities = self.architecture.forward(features, is_training=False)

            # This part of code is for ROCs
            probabilities_all = np.concatenate((probabilities_all, probabilities.astype(np.float32)))

            predictions = np.argmax(probabilities, axis=1)
            golden_classes = np.squeeze(golden_labels, axis=1)

            # This part of code is for ROCs
            golden_classes_all = np.concatenate((golden_classes_all, golden_classes.astype(np.int32)))

            well_predicted = predictions == golden_classes

            # This part of code is for plotting misclassified examples
            bad_predicted_indices = np.where(well_predicted == 0)

            for bad_predicted_index in bad_predicted_indices[0]:
                bad_predicted_list[int(golden_labels[bad_predicted_index])].append(features[bad_predicted_index]
                                                                                   .reshape([28, 28]))

            # This part of code is for counting metrics
            epoch_well_predicted += int(np.sum(well_predicted))
            epoch_all_examples += well_predicted.shape[0]

            golden_one_hot = convert_to_one_hot(golden_labels, class_number=probabilities.shape[1])
            avg_loss = self.loss.get_overall(golden_one_hot)
            epoch_loss_list.append(avg_loss)

        epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
        epoch_accuracy = epoch_well_predicted / epoch_all_examples

        return epoch_loss, epoch_accuracy, bad_predicted_list, golden_classes_all, probabilities_all








