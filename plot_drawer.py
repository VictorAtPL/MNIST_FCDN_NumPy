from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


class PlotDrawer:

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

        if len(labels) // 2 > 1:
            self.fig, self.axs = plt.subplots(len(labels) // 2, 1, sharex=True)
        else:
            self.fig, ax = plt.subplots()
            self.axs = [ax]

        self.xs = []
        self.ys = []

    def add_points(self, x: int, y: Tuple[float, ...]):
        if not self.plot_graphs:
            return

        self.xs.append(x)
        self.ys.append(y)

    def plot(self):
        if not self.plot_graphs or len(self.xs) == 0 or len(self.ys) == 0:
            return

        for ax in self.axs:
            # TODO: Could be plotted dynamically without clearing all plot
            ax.clear()

        self.axs[0].set_title("Losses values over epochs")
        self.axs[0].set_xlabel("epoch")
        self.axs[0].set_ylabel("loss")

        if len(self.axs) > 1:
            self.axs[1].set_title("Accuracies values over epochs")
            self.axs[1].set_xlabel("epoch")
            self.axs[1].set_ylabel("accuracy")

        for dim_no, dim_val in enumerate(zip(*self.ys)):
            self.axs[dim_no % 2].plot(self.xs, dim_val, label=self.labels[dim_no])

        for ax in self.axs:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        self.fig.legend()
        self.fig.show()
