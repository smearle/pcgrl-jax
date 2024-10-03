from functools import partial
import math
from typing import Tuple

import chex
from flax import struct
from gymnax.environments import spaces
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

    
@partial(jax.jit, static_argnames=('tile_enum', 'act_shape'))
def gen_agent_coords(frz_map: chex.Array, tile_enum: Tiles, 
                     act_shape: Tuple[int, int]):

    # TODO: If using larger action patches, ignore patches that are all frozen.
    
    if act_shape != (1, 1):
        # This is static. Not going to be recomputed/recompiled each function
        # call... right? D:
        # TODO: Factor this out, only needs to be called once.
        agent_coords = np.argwhere(np.ones(frz_map.shape, dtype=np.uint8))
                                    # size=math.prod(frz_map.shape))
        # Filter out coordinates so that agent with larger action shapes do 
        # minimal redundant builds (note that they may still make some 
        # overlapping builds near the edges).
        m, n = act_shape
        agent_coords = agent_coords[
            (agent_coords[:, 0] % m == 0) &
            (agent_coords[:, 1] % n == 0)]
        n_valid_agent_coords = np.int32(len(agent_coords))
    else:
        # Skip frozen tiles.
        agent_coords = jnp.argwhere(frz_map == 0, size=math.prod(frz_map.shape))
        n_valid_agent_coords = jnp.sum(frz_map == 0)
    return agent_coords, n_valid_agent_coords


class NarrowRepresentation(Representation):
    def __init__(self, env_map: chex.Array, rf_shape: Tuple[int, int],
                 act_shape: Tuple[int, int], tile_enum: Tiles, max_board_scans: int, pinpoints: bool, tile_nums: Tuple[int]
                 ):
        super().__init__(tile_enum=tile_enum, rf_shape=rf_shape,
                         act_shape=act_shape, pinpoints=pinpoints, tile_nums=tile_nums)
        self.rf_shape = np.array(rf_shape)
        self.rf_off = int(max(np.ceil(self.rf_shape - 1) / 2))
        self.max_steps = np.uint32(np.prod(env_map.shape) * max_board_scans)
        self.num_tiles = len(tile_enum)
        self.builds = jnp.array(self.editable_tile_enum)

        # agent_coords, self.n_valid_agent_coords = gen_agent_coords(
        #     env_map, tile_enum, act_shape)
        self.act_shape = act_shape

    def step(self, env_map: chex.Array, action: int,
             rep_state: NarrowRepresentationState, step_idx: int, agent_id: int = 0):
        action = action[..., 0]
        b = self.builds[action]
        pos_idx = step_idx % rep_state.n_valid_agent_coords
        new_pos = rep_state.agent_coords[pos_idx]
        new_env_map = jax.lax.dynamic_update_slice(env_map, b, new_pos)

        map_changed = jnp.logical_not(jnp.array_equal(new_env_map, env_map))
        rep_state = NarrowRepresentationState(
            pos=new_pos,
            agent_coords=rep_state.agent_coords,
            n_valid_agent_coords=rep_state.n_valid_agent_coords,
        )

        return new_env_map, map_changed, rep_state

    def reset(self, frz_map: chex.Array = None, rng: chex.PRNGKey = None):
        agent_coords, n_valid_agent_coords = gen_agent_coords(
            frz_map, self.tile_enum, self.act_shape)
        pos = agent_coords[0]

        return NarrowRepresentationState(
            pos=pos,
            agent_coords=agent_coords,
            n_valid_agent_coords=n_valid_agent_coords)
    
    @property
    def action_space(self) -> spaces.Discrete:
        # return spaces.Discrete(len(self.tile_enum) - 1)
        return spaces.Discrete(self.n_editable_tiles)

    get_obs = get_ego_obs
