from typing import Tuple

import chex
from flax import struct
import jax.numpy as jnp
import numpy as np

from envs.reps.representation import (Representation, RepresentationState,
                                      get_global_obs)
from envs.utils import Tiles


@struct.dataclass
class NCARepresentationState(RepresentationState):
    pass


class NCARepresentation(Representation):
    def __init__(self, env_map: chex.Array, rf_shape: Tuple[int, int],
                 act_shape: Tuple[int, int],
                 tile_enum: Tiles,
                 max_board_scans: int,
                 pinpoints: bool,
                 tile_nums: Tuple[int],
                 ):
        super().__init__(tile_enum=tile_enum, rf_shape=rf_shape,
                         act_shape=act_shape, pinpoints=pinpoints, tile_nums=tile_nums)
        self.env_map_shape = tuple(np.array(env_map.shape))
        # the idea being that this is enough time for activation to flow from any point on the map to the next
        # (assuming the NCA is allowed to change all cells at each step...)
        self.max_steps = np.uint32((env_map.shape[0] + env_map.shape[1]) * max_board_scans)
        self.num_tiles = np.uint32(len(tile_enum))
        self.tiles_enum = tile_enum

    def observation_shape(self):
        # Always observe static tile channel. Do not observe border tiles.
        return self.env_map_shape + (self.num_tiles, )

    def step(self, env_map: chex.Array, action: chex.Array,
             rep_state: NCARepresentationState, step_idx: int
             ):
        new_env_map = action[..., 0] + 1  # Exclude border tiles

        # map_changed = jnp.logical_not(jnp.array_equal(new_env_map, env_map))

        return new_env_map, rep_state

    def reset(self, static_map, rng):
        return NCARepresentationState()

    get_obs = get_global_obs
