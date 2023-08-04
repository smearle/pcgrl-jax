from typing import Tuple

import chex
from flax import struct
import jax
import jax.numpy as jnp

from envs.reps.representation import Representation, RepresentationState


@struct.dataclass
class NCARepresentationState(RepresentationState):
    pass


class NCARepresentation(Representation):
    def __init__(self, env_map: chex.Array, rf_size: int, num_tiles: int, tiles_enum):
        super().__init__(tiles_enum)
        self.rf_size = rf_size
        # self.rf_off = jnp.int32((self.rf_size - 1) / 2)
        # self.max_steps = (env_map.shape[0] - 2 * self.rf_off) + (env_map.shape[1] - 2 * self.rf_off)
        self.max_steps = (env_map.shape[0] + env_map.shape[1]) * 3
        self.num_tiles = num_tiles
        self.tiles_enum = tiles_enum
        # self.builds = jnp.array([tile for tile in tiles_enum if tile != tiles_enum.BORDER])
        self.agent_coords = jnp.argwhere(env_map != tiles_enum.BORDER)

    def observation_shape(self):
        return (self.rf_size, self.rf_size, self.num_tiles - 1)
        
    def step(self, env_map: chex.Array, action: chex.Array, rep_state: NCARepresentationState):
        new_env_map = action + 1 # Exclude border tiles

        map_changed = jnp.logical_not(jnp.array_equal(new_env_map, env_map))

        return new_env_map, map_changed, rep_state

    def reset(self):
        return NCARepresentationState()
        
    def get_obs(self, env_map: chex.Array, rep_state: NCARepresentationState):
        # Convert to one-hot encoding
        rf_obs = jax.nn.one_hot(env_map - 1, self.num_tiles - 1)
        return rf_obs