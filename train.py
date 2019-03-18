import argparse

from dataset import Dataset
from model import Model
from utils import setup_logger, get_logger


def train(args) -> None:
    logger = get_logger()

    logger.debug("Creating Dataset object")
    dataset = Dataset(args.data_dir)

    logger.debug("Creating model object")
    model = Model(args.batch_size, dataset, args.weight_decay_lambda, args.learning_rate)
    model.train(args.epochs, with_validation=args.train_with_validation, plot_graphs=args.plot_graphs)


def main() -> None:
    setup_logger()

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--weight_decay_lambda", default=0.2, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--train_with_validation", default=True, type=bool)
    parser.add_argument("--plot_graphs", default=False, type=bool)

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
