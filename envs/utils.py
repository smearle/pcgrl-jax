
from enum import IntEnum

import numpy as np


class Tiles(IntEnum):
    BORDER = 0
    EMPTY = 1


def idx_dict_to_arr(d):
    """Convert dictionary to array, where dictionary has form (index: value)."""
    return np.array([d[i] for i in range(len(d))])