from typing import List, Tuple, Any, Dict

import cupy
import cupy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve, auc

from utils import get_logger


class PlotDrawer:

    logger = get_logger()

    plot_graphs: bool

    labels: List[str]

    fig: Figure = None

    axs: List[Axes] = None

    xs: List[int] = None

    ys: List[Tuple[float, ...]] = None

    def __init__(self, plot_graphs: bool, labels=List[str]) -> None:
        self.plot_graphs = plot_graphs

        if not self.plot_graphs:
            return

        self.labels = labels

        self.fig, self.axs = plt.subplots(len(labels) // 2, 1, figsize=(8, 6))

        for ax in self.axs:
            box = ax.get_position()

            ax.set_position([box.x0, box.y0 + box.height * 0.3,
                             box.width, box.height * 0.7])

        self.xs = []
        self.ys = []

        self.fig.show()
        self.fig.canvas.draw()

    def add_points(self, x: int, y: Tuple[float, ...]) -> None:
        if not self.plot_graphs:
            return

        self.xs.append(x)
        self.ys.append(y)

    def plot(self) -> None:
        if not self.plot_graphs:
            return

        for ax in self.axs:
            # TODO: Could be plotted dynamically without clearing all plot
            ax.clear()

        self.axs[0].set_title("Losses values over epochs")
        self.axs[0].set_xlabel("epoch")
        self.axs[0].set_ylabel("loss")
        self.axs[0].grid()

        self.axs[1].set_title("Accuracies values over epochs")
        self.axs[1].set_xlabel("epoch")
        self.axs[1].set_ylabel("accuracy")
        self.axs[1].grid()

        for dim_no, dim_val in enumerate(zip(*self.ys)):
            self.axs[dim_no % 2].plot(self.xs, dim_val, label=self.labels[dim_no])

        for ax in self.axs:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
                      fancybox=True, shadow=True, ncol=2)

        self.fig.canvas.draw()

    def plot_wrong_classified(self, bad_predicted_list: Dict[Any, Any]) -> None:
        if not self.plot_graphs:
            return

        max_wrong_in_classes = max([len(bad_predicted) for bad_predicted in bad_predicted_list.values()])

        rows = 10
        cols = min(max_wrong_in_classes, 50)

        fig = plt.figure(figsize=[cols * 15.0/50.0, cols * 7.2/50.00])

        for row, bad_predicted in enumerate(bad_predicted_list.values()):
            for col, example in enumerate(bad_predicted, start=1):
                if col > cols:
                    break

                ax = fig.add_subplot(rows, cols, row * cols + col)

                if col == 1:
                    ax.set_ylabel("class " + str(row) + "\n" + str(len(bad_predicted)), rotation=0, fontsize=10)
                    ax.set_yticklabels([])
                    ax.get_yaxis().set_ticks([])
                    ax.get_yaxis().set_label_coords(-2, -0.5)
                else:
                    ax.get_yaxis().set_visible(False)

                ax.imshow(example.get())
                ax.get_xaxis().set_visible(False)

        fig.suptitle("Wrongly classified examples (max 50 per class)")

        fig.show()

    def plot_rocs(self, golden_classes: np.ndarray, probabilities: np.ndarray) -> None:
        if not self.plot_graphs:
            return

        n_classes = np.unique(golden_classes).shape[0]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        fig, axs = plt.subplots(5, 2, figsize=(15, 25))
        fig.subplots_adjust(hspace=0.5, wspace=0.1)
        axs = axs.flatten()

        lw = 2
        for i in range(n_classes):
            y_class = np.where(golden_classes == i, 1, 0)
            y_score = probabilities[:, i]

            if isinstance(y_class, cupy.ndarray):
                y_class = cupy.asnumpy(y_class)
                y_score = cupy.asnumpy(y_score)

            fpr[i], tpr[i], _ = roc_curve(y_class, y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])

            axs[i].plot(fpr[i], tpr[i], color="darkorange",
                        lw=lw, label="ROC curve (area = %0.3f)" % roc_auc[i])
            axs[i].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            axs[i].set_xlim([0.0, 1.0])
            axs[i].set_ylim([0.0, 1.05])
            axs[i].set_xlabel("False Positive Rate")
            axs[i].set_ylabel("True Positive Rate")
            axs[i].legend(loc="lower right", fancybox=True, shadow=True)
            axs[i].set_title("Class " + str(i))
            axs[i].set_aspect("equal")

        fig.suptitle("Receiver operating characteristic")
        fig.show()





