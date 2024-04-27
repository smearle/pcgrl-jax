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
        self.directions = jnp.array([[0, 0] for _ in range(
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
        #print("action is: ", action)
        #print(jnp.array(self.directions)[action])
        #deltas = jnp.array(self.directions)[action].sum((0, 1))
        new_pos = a_pos + self.directions[action]
        new_pos = jnp.clip(
            new_pos,
            np.array((0, 0)),
            np.array((env_map.shape[0] - 1, env_map.shape[1] - 1))
        )
        can_move = env_map[new_pos[0], new_pos[1]] != self.tile_enum.BORDER
        new_pos = jax.lax.select(can_move, new_pos, a_pos)

        # Meaningless if agent is moving.
        build = self.builds[action].reshape((1,1))
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
                 n_agents: int, max_board_scans: float, pinpoints: bool, tile_nums: Tuple[int]):
        super().__init__(tile_enum=tile_enum, rf_shape=rf_shape,
                         act_shape=act_shape, env_map=env_map, map_shape=map_shape,
                         max_board_scans=max_board_scans, pinpoints=pinpoints, tile_nums=tile_nums)
        self.map_shape = map_shape
        self.max_steps = int(math.ceil(self.max_steps / n_agents))
        self.n_agents = n_agents
        self.act_coords = np.argwhere(np.ones(map_shape))



    def observation_shape(self):
        # Always observe static tile channel, agent location channel
        # return (*self.rf_shape, self.n_agents * (len(self.tile_enum) + 1 + self.n_agents)) commented out, no difference inbetween other agents
        return {'sample_agent': (*self.rf_shape, len(self.tile_enum) + 1)}



    def step(self, env_map: chex.Array, action: int, step_idx: int,
             rep_state: MultiTurtleRepresentationState):     
        
        def single_step(carry, x):
            # unpack both carry and x
            new_env_map, map_changed = carry
            a_pos, a_action = x

            new_env_map, a_map_changed, new_a_pos = self.step_turtle(new_env_map, a_action, a_pos)
            map_changed = jnp.logical_or(map_changed, a_map_changed)
        
            carry = new_env_map, map_changed
            return carry, new_a_pos

        carry, new_positions = jax.lax.scan(
                single_step, 
                (env_map, False), #carry: new_env_map, map_changed 
                (rep_state.pos, action) #xs: a_pos, a_action
                )

        rep_state = rep_state.replace(pos=new_positions)
        # returns new_env_map, map_changed, rep_state
        return *carry, rep_state



    def reset(self, frz_map, rng):
        # randomly create n_agents starting positions
        shuffled_indices = jax.random.permutation(rng, self.act_coords, independent=True)
        return TurtleRepresentationState(pos=shuffled_indices[:self.n_agents])



    def get_obs(self, env_map: chex.Array, static_map: chex.Array,
                rep_state: RepresentationState):
     
        # will add 1 where a single agent is and scan over it
        def single_update(agents_layer, pos):
            agents_layer = agents_layer.at[*pos].add(1)
            return agents_layer, 0

        agents_layer, _ = jax.lax.scan(single_update, jnp.zeros(env_map.shape), rep_state.pos)
       
        # normalizes observations to force between 0-1 and pad
        agents_layer = agents_layer / self.n_agents
        agents_layer = jnp.pad(agents_layer,
                self.rf_off,
                mode='constant',
                constant_values=self.tile_enum.BORDER
                )

        # get board observation for a single agent and concatenate the other agent plane
        def single_obs(agent_pos):
            ego_observation = get_ego_obs(self, env_map, static_map,
                                          TurtleRepresentationState(agent_pos))
           
            other_agent_layer = jax.lax.dynamic_slice(
                    agents_layer,
                    agent_pos,
                    self.rf_shape
                    )

            ego_observation = jnp.concatenate([ego_observation, other_agent_layer[..., None]], axis=-1)
            return ego_observation
        
        # creates the dictionary for the agents' observations
        observations = {f"agent{i}":single_obs(agent_pos) for i, agent_pos in enumerate(rep_state.pos)}
        return observations

