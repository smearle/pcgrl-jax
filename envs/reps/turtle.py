from flax import struct
from typing import Tuple
import chex
from gymnax.environments import spaces
import jax
import jax.numpy as jnp
from envs.reps.representation import (Representation, RepresentationState,
                                      get_ego_obs)
from envs.utils import Tiles


@struct.dataclass
class TurtleRepresentationState(RepresentationState):
    pos: Tuple[int, int]


class TurtleRepresentation(Representation):
    def __init__(self, env_map: chex.Array, rf_shape: Tuple[int, int],
                 tile_enum: Tiles, act_shape: Tuple[int, int],
                 ):
        super().__init__(tile_enum=tile_enum, rf_shape=rf_shape,
                         act_shape=act_shape)
        self.rf_size = rf_shape
        self.rf_off = jnp.int32((self.rf_size - 1) / 2)
        self.max_steps = (
            env_map.shape[0] - 2 * self.rf_off) * (env_map.shape[1]
                                                   - 2 * self.rf_off) * 2
        self.num_tiles = len(tile_enum)
        self.tile_enum = tile_enum
        self.directions = jnp.array([[0, 0] for _ in range(
            len(self.num_tiles) - 1)] + [[-1, 0], [0, 1], [1, 0], [0, -1]])
        self.builds = jnp.array(
            [tile for tile in tile_enum if tile != tile_enum.BORDER] + [0] * 4
        )
        center = jnp.int32((env_map.shape[0] - 1) / 2)
        self.center_position = jnp.array([center, center])

    def step(self, env_map: chex.Array, action: int,
             rep_state: TurtleRepresentationState):
        p = rep_state.pos + self.directions[action]
        in_map = env_map[p[0], p[1]] != self.tile_enum.BORDER
        new_pos = jax.lax.select(in_map, p, rep_state.pos)

        # Sample a new starting position for case when goal is reached
        b = self.builds[action]  # Meaningless if agent is moving.
        nem = env_map.at[new_pos[0], new_pos[1]].set(jnp.array(b, int))
        valid_build = b != 0
        new_env_map = jax.lax.select(valid_build, nem, env_map)

        # Update state dict and evaluate termination conditions
        map_changed = jnp.logical_and(valid_build, jnp.logical_not(
            jnp.array_equal(new_env_map, env_map)))
        rep_state = rep_state.replace(pos=new_pos)

        return new_env_map, map_changed, rep_state

    def reset(self):
        return TurtleRepresentationState(pos=self.center_position)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.tile_enum) + 4)

    get_obs = get_ego_obs
