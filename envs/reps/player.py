import math
from typing import Tuple

from flax import struct
import chex
from gymnax.environments import spaces
import jax

from jax.experimental import checkify
import jax.numpy as jnp

import numpy as np
from envs.reps.representation import (Representation, RepresentationState,
                                      get_ego_obs)
from envs.utils import Tiles


@struct.dataclass
class PlayerRepresentationState(RepresentationState):
    pos: Tuple[int, int]


class PlayerRepresentation(Representation):
    def __init__(self, env_map: chex.Array, rf_shape: Tuple[int, int],
                 tile_enum: Tiles, act_shape: Tuple[int, int], map_shape: Tuple[int, int],
                 max_board_scans: float = 3.0
                 ):
        super().__init__(tile_enum=tile_enum, rf_shape=rf_shape,
                         act_shape=act_shape)
        self.rf_shape = np.array(rf_shape)
        self.rf_off = int(max(np.ceil(self.rf_shape - 1) / 2))
        self.max_steps = np.uint32((env_map.shape[0] * env_map.shape[1]) * 2 * max_board_scans)
        self.num_tiles = len(tile_enum)
        self.tile_enum = tile_enum
        self.directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        center = jnp.int32((env_map.shape[0] - 1) / 2)
        self.center_position = jnp.array([center, center])

    @property
    def tile_action_dim(self):
        return 4

    def step(self, env_map: chex.Array, action: chex.Array,
             rep_state: PlayerRepresentationState, step_idx: int):
        a_pos = rep_state.pos
        action = action[..., 0]
        # Sum directions over tilewise actions (jumping turtle).
        deltas = jnp.array(self.directions)[action].sum((0, 1))
        new_pos = a_pos + deltas
        new_pos = jnp.clip(
            new_pos,
            np.array((0, 0)),
            np.array((env_map.shape[0] - 1, env_map.shape[1] - 1))
        )
        can_move = jnp.all(env_map[tuple(new_pos)] != 
                           np.array([self.tile_enum.BORDER, self.tile_enum.WALL]))
        new_pos = jax.lax.select(can_move, new_pos, a_pos)

        # env_map = env_map.at[tuple(a_pos)].set(self.tile_enum.EMPTY)
        # env_map = jax.lax.select(
        #     env_map.at[tuple(new_pos)] != self.tile_enum.DOOR,
        #     env_map.at[tuple(new_pos)].set(self.tile_enum.PLAYER),
        #     env_map.at[tuple(a_pos)].set(self.tile_enum.PLAYER)
        # )
        # new_env_map = jax.lax.dynamic_update_slice(
        #     env_map,
        #     jnp.array(self.tile_enum.PLAYER)[None,None],
        #     new_pos
        # )
        # new_env_map = jnp.where(new_env_map != -1, new_env_map, env_map)

        # Never overwrite border
        # new_env_map = jnp.where(env_map != self.tile_enum.BORDER,
        #                         new_env_map, env_map)

        # Update state dict and evaluate termination conditions
        # map_changed = jnp.logical_not(jnp.array_equal(new_env_map, env_map))

        rep_state = rep_state.replace(pos=new_pos)
        map_changed = True
        return env_map, map_changed, rep_state

    def reset(self, static_tiles: chex.Array, rng: chex.PRNGKey):
        return PlayerRepresentationState(pos=self.center_position)

    def action_space(self) -> spaces.Discrete:
        # Cannot build border tiles. Can move in 4 directions.
        return spaces.Discrete(self.tile_action_dim)

    get_obs = get_ego_obs


