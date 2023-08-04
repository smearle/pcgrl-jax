from flax import struct
from typing import Tuple
import chex
from gymnax.environments import spaces
import jax
import jax.numpy as jnp
from envs.reps.representation import Representation, RepresentationState, get_ego_obs


@struct.dataclass
class TurtleRepresentationState(RepresentationState):
    pos: Tuple[int, int]


class TurtleRepresentation(Representation):
    def __init__(self, env_map: chex.Array, rf_size, num_tiles, tiles_enum):
        self.rf_size = rf_size
        self.rf_off = jnp.int32((self.rf_size - 1) / 2)
        self.max_steps = (env_map.shape[0] - 2 * self.rf_off) * (env_map.shape[1] - 2 * self.rf_off) * 2
        self.num_tiles = num_tiles
        self.tiles_enum = tiles_enum
        self.directions = jnp.array([[0, 0] for _ in range(len(self.num_tiles) - 1)] + [[-1, 0], [0, 1], [1, 0], [0, -1]])
        # self.builds = jnp.array([Tiles.EMPTY, Tiles.WALL, 0, 0, 0, 0])
        self.builds = jnp.array([tile for tile in tiles_enum if tile != tiles_enum.BORDER] + [0] * 4)
        # center = jnp.int32((self.env_map.shape[0] - 1) / 2 + self.rf_off - 1)
        center = jnp.int32((env_map.shape[0] - 1) / 2)
        self.center_position = jnp.array([center, center])

    def step(self, env_map: chex.Array, action: int, rep_state: TurtleRepresentationState):
        p = rep_state.pos + self.directions[action]
        in_map = env_map[p[0], p[1]] != self.tiles_enum.BORDER
        new_pos = jax.lax.select(in_map, p, rep_state.pos)
        # goal_reached = jnp.logical_and(
        #     new_pos[0] == state.goal[0], new_pos[1] == state.goal[1]
        # )
        # reward = (
        #     goal_reached * params.reward  # Add goal reward
        #     # + (1 - in_map) * params.punishment  # Add punishment for wall
        # )

        # Sample a new starting position for case when goal is reached
        # pos_sampled = reset_pos(key, self.coords)
        # new_pos = jax.lax.select(goal_reached, pos_sampled, new_pos)
        b = self.builds[action]  # Meaningless if agent is moving.
        nem = env_map.at[new_pos[0], new_pos[1]].set(jnp.array(b, int))
        # If agent isn't moving, then let it build
        # valid_build = jnp.logical_and(b != 0, state.env_map[new_pos[0], new_pos[1]] != Tiles.GOAL)
        valid_build = b != 0
        new_env_map = jax.lax.select(valid_build, nem, env_map)

        # Update state dict and evaluate termination conditions
        # reward = new_occ_map.sum() - state.occupied_map.sum()
        # new_path_length, flood_state = self.calc_path(state, new_occ_map)
        map_changed = jnp.logical_and(valid_build, jnp.logical_not(jnp.array_equal(new_env_map, env_map)))
        rep_state = rep_state.replace(pos=new_pos)

        return new_env_map, map_changed, rep_state

    def reset(self):
        return TurtleRepresentationState(pos=self.center_position)
        
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.tiles_enum) + 4)

    get_obs = get_ego_obs