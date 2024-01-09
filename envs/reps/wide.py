import math
from typing import Tuple

import chex
from flax import struct
from gymnax.environments import spaces
import jax.numpy as jnp
import numpy as np

from envs.reps.representation import (Representation, RepresentationState,
                                      get_global_obs)
from envs.utils import Tiles


@struct.dataclass
class WideRepresentationState(RepresentationState):
    pos: Tuple[int, int]


class WideRepresentation(Representation):
    def __init__(self, env_map: chex.Array, rf_shape: Tuple[int, int],
                 act_shape: Tuple[int, int], tile_enum: Tiles,
                 max_board_scans: float,
                 ):
        super().__init__(tile_enum=tile_enum, rf_shape=rf_shape,
                         act_shape=act_shape
                         )
        self.rf_shape = np.array(env_map.shape)
        self.rf_off = int(max(np.ceil(self.rf_shape - 1) / 2))
        self.max_steps = np.uint32(env_map.shape[0] * env_map.shape[1] * max_board_scans)
        self.num_tiles = len(tile_enum)
        self.map_shape = (*env_map.shape, self.num_tiles)
        self.builds = jnp.array(
            [tile for tile in tile_enum if tile != tile_enum.BORDER])

    def observation_shape(self):
        # Always observe static tile channel. Do not observe border tiles.
        return (*self.rf_shape, len(self.tile_enum))

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete((len(self.tile_enum) - 1) *
                               math.prod(self.map_shape))

    def step(self, env_map: chex.Array, action: chex.Array,
             rep_state: WideRepresentationState, step_idx: int):
        action = action[0,0]  # The x and y dimensions are ignored
        action = jnp.unravel_index(action, self.map_shape)
        x, y, b = action
        b = self.builds[b]
        new_env_map = env_map.at[x, y].set(b)
        map_changed = jnp.logical_not(jnp.array_equal(new_env_map, env_map))
        pos = jnp.concatenate((x, y))
        rep_state = WideRepresentationState(pos=pos)

        return new_env_map, map_changed, rep_state

    def reset(self, static_tiles: chex.Array = None, rng: chex.PRNGKey = None):
        return WideRepresentationState(pos=jnp.array((0, 0)))

    get_obs = get_global_obs
