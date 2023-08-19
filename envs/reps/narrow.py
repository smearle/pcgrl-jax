from typing import Tuple

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from envs.utils import Tiles
from envs.reps.representation import (Representation, RepresentationState,
                                      get_ego_obs)


@struct.dataclass
class NarrowRepresentationState(RepresentationState):
    pos: chex.Array
    agent_coords: chex.Array
    n_valid_agent_coords: int


class NarrowRepresentation(Representation):
    def __init__(self, env_map: chex.Array, rf_shape: Tuple[int, int],
                 act_shape: Tuple[int, int], tile_enum: Tiles
                 ):
        super().__init__(tile_enum=tile_enum, rf_shape=rf_shape,
                         act_shape=act_shape
                         )
        self.rf_shape = np.array(rf_shape)
        self.rf_off = int(max(np.ceil(self.rf_shape - 1) / 2))
        self.max_steps = np.int32(env_map.shape[0] * env_map.shape[1])
        self.num_tiles = len(tile_enum)
        self.builds = jnp.array(
            [tile for tile in tile_enum if tile != tile_enum.BORDER])
        self.agent_coords = jnp.argwhere(env_map != tile_enum.BORDER)
        self.n_valid_agent_coords = np.int32(len(self.agent_coords))
        self.act_shape = act_shape

    def step(self, env_map: chex.Array, action: int,
             rep_state: NarrowRepresentationState, step_idx: int):
        b = self.builds[action]
        pos_idx = step_idx % self.n_valid_agent_coords
        new_pos = self.agent_coords[pos_idx]
        # new_env_map = env_map.at[new_pos[0], new_pos[1]].set(b[0])
        new_env_map = jax.lax.dynamic_update_slice(env_map, b[0], new_pos)

        map_changed = jnp.logical_not(jnp.array_equal(new_env_map, env_map))
        rep_state = NarrowRepresentationState(
            pos=new_pos,
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
        return NarrowRepresentationState(
            pos=pos,
            agent_coords=agent_coords,
            n_valid_agent_coords=n_valid_agent_coords)

    get_obs = get_ego_obs
