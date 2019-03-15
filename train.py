import argparse

from dataset import Dataset
from model import Model
from utils import setup_logger, get_logger


def train(args) -> None:
    logger = get_logger()

    logger.debug("Creating Dataset object")
    dataset = Dataset(args.data_dir)

    logger.debug("Creating model object")
    model = Model(32, dataset)
    model.train()


def main() -> None:
    setup_logger()

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/")
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
