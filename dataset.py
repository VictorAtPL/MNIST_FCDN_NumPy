import os
import struct
from typing import Generator, List, Dict, Tuple

import cupy as np

from dataset_type import DatasetType
from utils import get_logger


class Dataset:

    logger = get_logger()

    FILE_NAMES: Dict[str, Dict[DatasetType, str]] = {
        "images": {
            DatasetType.TRAINING: "train-images-idx3-ubyte",
            DatasetType.VALIDATION: "t10k-images-idx3-ubyte"
        },
        "labels": {
            DatasetType.TRAINING: "train-labels-idx1-ubyte",
            DatasetType.VALIDATION: "t10k-labels-idx1-ubyte"
        }
    }

    cache: Dict[DatasetType, Tuple[np.ndarray, np.ndarray]] = {}

    def __init__(self, data_dir: str) -> None:
        super().__init__()

        self.data_dir = data_dir

    def get_batch(self, dataset_type: DatasetType, batch_size: int = 32) \
            -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if dataset_type not in self.cache:
            self.cache[dataset_type] = self._get_subset(dataset_type)

        number_of_examples = len(self.cache[dataset_type][0])
        indices = np.random.permutation(number_of_examples)

        for batch_number in range(number_of_examples // batch_size):
            slice_start = batch_number * batch_size
            slice_end = slice_start + batch_size
            slice_indices = indices[slice_start:slice_end]
            yield self.cache[dataset_type][0].take(slice_indices, axis=0),\
                  self.cache[dataset_type][1].take(slice_indices, axis=0),

        if number_of_examples % batch_size != 0:
            slice_start = number_of_examples // batch_size * batch_size
            slice_end = None
            slice_indices = indices[slice_start:slice_end]
            yield self.cache[dataset_type][0].take(slice_indices, axis=0), \
                  self.cache[dataset_type][1].take(slice_indices, axis=0),

    def _get_subset(self, dataset_type: DatasetType) -> Tuple[np.ndarray, np.ndarray]:
        subset = self._read_images_file(self.FILE_NAMES["images"][dataset_type]), \
                 self._read_labels_file(self.FILE_NAMES["labels"][dataset_type])

        self.logger.debug("Loaded %s set", dataset_type)
        return subset

    def _read_labels_file(self, file_name: str):
        """

        :param file_name:
        :return: array of shape (examples_no)
        """
        byte_generator = self._bytes_from_file(os.path.join(self.data_dir, file_name))

        _ = struct.unpack(">i", self._get_next_n_bytes(byte_generator))[0]
        number_of_items = struct.unpack(">i", self._get_next_n_bytes(byte_generator))[0]

        labels_list = [self._get_next_n_ints(byte_generator, number_of_items)]

        labels_arr = np.array(labels_list, dtype=np.int32).reshape([number_of_items, 1])

        self.logger.debug("Loaded labels from %s file", file_name)
        return labels_arr

    def _read_images_file(self, file_name: str) -> np.ndarray:
        """

        :param file_name:
        :return: array of shape (examples_no, image_rows, image_cols)
        """
        byte_generator = self._bytes_from_file(os.path.join(self.data_dir, file_name))

        _ = struct.unpack(">i", self._get_next_n_bytes(byte_generator))[0]
        number_of_images = struct.unpack(">i", self._get_next_n_bytes(byte_generator))[0]
        number_of_rows = struct.unpack(">i", self._get_next_n_bytes(byte_generator))[0]
        number_of_columns = struct.unpack(">i", self._get_next_n_bytes(byte_generator))[0]

        number_of_pixels = number_of_rows * number_of_columns

        images_list = [self._get_next_n_ints(byte_generator, number_of_pixels) for _ in range(number_of_images)]

        images_arr = np.array(images_list, dtype=np.float)
        self.logger.debug("Loaded images from %s file", file_name)
        return images_arr

    @staticmethod
    def _get_next_n_ints(generator: Generator[int, None, None], n: int = 4) -> List[int]:
        return [next(generator) for _ in range(n)]

    @staticmethod
    def _get_next_n_bytes(generator: Generator[int, None, None], n: int = 4) -> bytes:
        return bytes(Dataset._get_next_n_ints(generator, n))

    @staticmethod
    def _bytes_from_file(filename: str, chunk_size: int = 8192) -> Generator[int, None, None]:
        """From https://stackoverflow.com/a/1035456/1625856"""

        with open(filename, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if chunk:
                    for b in chunk:
                        yield b
                else:
                    break
