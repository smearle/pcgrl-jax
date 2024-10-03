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
class TurtleRepresentationState(RepresentationState):
    pos: Tuple[int, int]


class TurtleRepresentation(Representation):
    def __init__(self, env_map: chex.Array, rf_shape: Tuple[int, int],
                 tile_enum: Tiles, act_shape: Tuple[int, int], map_shape: Tuple[int, int],
                 max_board_scans: float, pinpoints: bool, tile_nums: Tuple[int],
                 ):
        super().__init__(tile_enum=tile_enum, rf_shape=rf_shape,
                         act_shape=act_shape, pinpoints=pinpoints, tile_nums=tile_nums)
        self.rf_shape = np.array(rf_shape)
        self.rf_off = int(max(np.ceil(self.rf_shape - 1) / 2))
        self.max_steps = np.uint32((env_map.shape[0] * env_map.shape[1]) * 2 * max_board_scans)
        self.num_tiles = len(tile_enum)
        self.directions = np.array([[0, 0] for _ in range(
            self.num_tiles - 1)] + [[-1, 0], [0, 1], [1, 0], [0, -1]])
        self.builds = jnp.array(
            self.editable_tile_enum + [-1] * 4
        )
        center = jnp.int32((env_map.shape[0] - 1) / 2)
        self.center_position = jnp.array([center, center])

    @property
    def tile_action_dim(self):
        return self.n_editable_tiles + 4

    def step(self, env_map: chex.Array, action: chex.Array,
             rep_state: TurtleRepresentationState, step_idx: int):
        new_env_map, map_changed, new_pos = self.step_turtle(env_map, action, rep_state.pos)
        rep_state = rep_state.replace(pos=new_pos)
        return new_env_map, map_changed, rep_state

    def step_turtle(self, env_map, action, a_pos):
        action = action[..., 0]
        # Sum directions over tilewise actions (jumping turtle).
        deltas = jnp.array(self.directions)[action].sum((0, 1))
        new_pos = a_pos + deltas
        new_pos = jnp.clip(
            new_pos,
            np.array((0, 0)),
            np.array((env_map.shape[0] - 1, env_map.shape[1] - 1))
        )
        can_move = env_map[new_pos[0], new_pos[1]] != self.tile_enum.BORDER
        new_pos = jax.lax.select(can_move, new_pos, a_pos)

        # Meaningless if agent is moving.
        build = jnp.array(self.builds)[action]
        # nem = env_map.at[new_pos[0], new_pos[1]].set(jnp.array(b, int))
        new_env_map = jax.lax.dynamic_update_slice(env_map, build, new_pos)
        new_env_map = jnp.where(new_env_map != -1, new_env_map, env_map)

        # Never overwrite border
        new_env_map = jnp.where(env_map != self.tile_enum.BORDER,
                                new_env_map, env_map)

        # Update state dict and evaluate termination conditions
        map_changed = jnp.logical_not(jnp.array_equal(new_env_map, env_map))

        return new_env_map, map_changed, new_pos

    def reset(self, static_tiles: chex.Array, rng: chex.PRNGKey):
        return TurtleRepresentationState(pos=self.center_position)

    @property
    def action_space(self) -> spaces.Discrete:
        # Cannot build border tiles. Can move in 4 directions.
        return spaces.Discrete(self.n_editable_tiles + 4)

    get_obs = get_ego_obs


@struct.dataclass
class MultiTurtleRepresentationState(RepresentationState):
    pos: Tuple[Tuple[int, int]]


class MultiTurtleRepresentation(TurtleRepresentation):
    def __init__(self, env_map: chex.Array, rf_shape: Tuple[int, int],
                 tile_enum: Tiles, act_shape: Tuple[int, int], map_shape: Tuple[int, int],
                 n_agents: int, max_board_scans: float, pinpoints: bool, tile_nums: Tuple[int],):
        super().__init__(tile_enum=tile_enum, rf_shape=rf_shape,
                         act_shape=act_shape, env_map=env_map, map_shape=map_shape,
                         max_board_scans=max_board_scans, pinpoints=pinpoints, tile_nums=tile_nums)
        self.map_shape = map_shape
        self.max_steps = int(math.ceil(self.max_steps / n_agents))
        self.n_agents = int(n_agents)
        self.act_coords = np.argwhere(np.ones(map_shape))

    def observation_shape(self):
        # Always observe static tile channel, agent location channel
        # return (*self.rf_shape, self.n_agents * (len(self.tile_enum) + 1 + self.n_agents))
        return (*self.rf_shape, len(self.tile_enum) + 1 + self.n_agents)

    def step(self, env_map: chex.Array, action: int, step_idx: int,
             rep_state: MultiTurtleRepresentationState, agent_id: int):

        map_changed = False
        new_env_map = env_map
        new_positions = rep_state.pos

        # for i, a_pos in enumerate(rep_state.pos):
        a_pos = rep_state.pos[agent_id]

        new_env_map, a_map_changed, new_a_pos = self.step_turtle(
            new_env_map, action, a_pos)
        map_changed = jnp.logical_or(map_changed, a_map_changed)

        new_positions = new_positions.at[agent_id].set(new_a_pos)

        rep_state = rep_state.replace(pos=new_positions)

        return new_env_map, map_changed, rep_state

    def reset(self, frz_map, rng):
        # Get all indices of board positions
        # shuffle
        # shuffled_indices = jax.random.shuffle(rng, self.act_coords)
        shuffled_indices = jax.random.permutation(rng, self.act_coords, independent=True)
        return TurtleRepresentationState(pos=shuffled_indices[:self.n_agents])
        # return TurtleRepresentationState(pos=jnp.repeat(self.center_position[None], self.n_agents,
                                                        # axis=0))

    def get_obs(self, env_map: chex.Array, static_map: chex.Array,
                rep_state: RepresentationState):
        padded_env_map = jnp.pad(
            env_map, self.rf_off, mode='constant',
            constant_values=self.tile_enum.BORDER)
        # Each agent is associated with a static tile channel, and a onehot agentlocation channel
        rf_obs = jnp.zeros((self.n_agents, *self.rf_shape, len(self.tile_enum) + 1 + self.n_agents))
        agent_loc_map = jnp.zeros(env_map.shape + (self.n_agents,))

        def set_agent_loc_map(carry, _):
            agent_loc_map, i = carry
            a_pos = rep_state.pos[i]
            return (agent_loc_map.at[a_pos[0], a_pos[1], i].set(1), i+1), None

        (agent_loc_map, _), _ = jax.lax.scan(set_agent_loc_map, (agent_loc_map, 0), None,
                     length=self.n_agents)

        # for i, a_pos in enumerate(rep_state.pos):
        #     agent_loc_map = agent_loc_map.at[a_pos[0], a_pos[1], i].set(1)

        padded_agent_loc_map = jnp.pad(
            agent_loc_map, 
            ((self.rf_off, self.rf_off), (self.rf_off, self.rf_off), (0, 0)),
            mode='constant',
            constant_values=((0, 0), (0, 0), (0, 0)))

        if static_map is not None:
            padded_static_map = jnp.pad(static_map, self.rf_off, mode='constant',
                                        constant_values=1)  # Border is static

        # TODO: Currently doing this too much! I.e. getting both agents' observations for *each* agent. Is redundant!!

        # Iterating through agents
        # for i, a_pos in enumerate(rep_state.pos):
        for i in range(self.n_agents):
            a_pos = rep_state.pos[i]
            rf_map_obs = jax.lax.dynamic_slice(
                padded_env_map,
                a_pos,
                self.rf_shape,
            )
            # Convert to one-hot encoding
            a_rf_obs = jax.nn.one_hot(rf_map_obs, self.num_tiles)
            if static_map is not None:
                a_rf_static_obs = jax.lax.dynamic_slice(
                    padded_static_map,
                    a_pos,
                    self.rf_shape,
                )
                a_rf_obs = jnp.concatenate(
                    [a_rf_obs, a_rf_static_obs[..., None]], axis=-1)

            oh_rf_shape = (*self.rf_shape, self.n_agents)
            oh_a_pos = jnp.concatenate([a_pos, jnp.array([0])])
            a_loc_map = jax.lax.dynamic_slice(
                padded_agent_loc_map,
                oh_a_pos,
                oh_rf_shape,
            )

            a_rf_obs = jnp.concatenate(
                [a_rf_obs, a_loc_map], axis=-1)

            rf_obs = rf_obs.at[i].set(a_rf_obs)

        # Collapse agent dimension (0) into channel dimension (-1)
        # rf_obs = jnp.concatenate(rf_obs, axis=-1)
        return rf_obs
