from typing import Tuple
import chex
import jax
import jax.numpy as jnp
import numpy as np
from envs.reps.representation import Representation, RepresentationState

from flax import struct

from envs.utils import Tiles

@struct.dataclass
class NarrowRepresentationState(RepresentationState):
    pos: Tuple[int, int]
    agent_coords: chex.Array
    n_valid_agent_coords: int


class NarrowRepresentation(Representation):
    def __init__(self, env_map: chex.Array, rf_shape: Tuple[int, int], tile_enum: Tiles):
        super().__init__(tile_enum=tile_enum)
        self.rf_shape = np.array(rf_shape)
        self.rf_off = int(max(np.ceil(self.rf_shape - 1) / 2))
        self.max_steps = np.int32(env_map.shape[0] * env_map.shape[1])
        # self.max_steps = (env_map.shape[0] - self.rf_off) * (env_map.shape[1] - self.rf_off)
        self.num_tiles = len(tile_enum)
        self.builds = jnp.array([tile for tile in tile_enum if tile != tile_enum.BORDER])
        self.agent_coords = jnp.argwhere(env_map != tile_enum.BORDER)
        self.n_valid_agent_coords = np.int32(len(self.agent_coords))

    def step(self, env_map: chex.Array, action: chex.Array, rep_state: NarrowRepresentationState, step_idx: int):
        pos_idx = (step_idx) % rep_state.n_valid_agent_coords
        new_pos = rep_state.agent_coords[pos_idx]
        b = self.builds[action] 
        new_env_map = env_map.at[new_pos[0], new_pos[1]].set(b[0])

        map_changed = jnp.logical_not(jnp.array_equal(new_env_map, env_map))
        rep_state = NarrowRepresentationState(
            pos = new_pos,
            agent_coords=rep_state.agent_coords,
            n_valid_agent_coords=rep_state.n_valid_agent_coords,
        )

        return new_env_map, map_changed, rep_state

    def reset(self, static_tiles: chex.Array = None):
        if static_tiles is not None:
            agent_coords = jnp.argwhere(static_tiles == 0, size=self.max_steps)
            n_valid_agent_coords = jnp.sum(static_tiles == 0)
        else:
            agent_coords = self.agent_coords
            n_valid_agent_coords = self.n_valid_agent_coords
        pos = self.agent_coords[0]
        return NarrowRepresentationState(pos=pos, agent_coords=agent_coords, n_valid_agent_coords=n_valid_agent_coords)
        
    def get_obs(self, env_map: chex.Array, rep_state: NarrowRepresentationState):
        padded_env_map = jnp.pad(env_map, self.rf_off, mode='constant', constant_values=self.tile_enum.BORDER)
        rf_obs = jax.lax.dynamic_slice(
            padded_env_map,
            rep_state.pos - self.rf_off,
            self.rf_shape,
        )
        # Convert to one-hot encoding
        rf_obs = jax.nn.one_hot(rf_obs, self.num_tiles)
        return rf_obs