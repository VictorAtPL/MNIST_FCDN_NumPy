from typing import Optional

import numpy as np


def convert_to_one_hot(tensor: np.ndarray, class_number: Optional[int] = None):
    assert len(tensor.shape) == 2 and tensor.shape[1] == 1

    if not class_number:
        unique_values_count = len(np.unique(tensor))
        labels_matrix = np.arange(unique_values_count).reshape([1, unique_values_count])
    else:
        labels_matrix = np.arange(class_number).reshape([1, class_number])

    one_hot = labels_matrix == tensor
    return one_hot.astype(np.float32)
