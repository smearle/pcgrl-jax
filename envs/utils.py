
from enum import IntEnum

import numpy as np
import jax.numpy as jnp


class Tiles(IntEnum):
    BORDER = 0
    EMPTY = 1


def idx_dict_to_arr(d):
    """Convert dictionary to array, where dictionary has form (index: value)."""
    return np.array([d[i] for i in range(len(d))])

def pad_env_map(env_map, mask_shape):
    """Pad map with zeros for use with mask."""
    padding = [(0, s) for s in mask_shape]
    return jnp.pad(env_map, padding)