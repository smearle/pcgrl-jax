from abc import ABC
import math
from typing import Tuple
import copy

import chex
from flax import struct
from gymnax.environments import spaces
import jax
import jax.numpy as jnp

from envs.utils import Tiles
from .representation import Representation, RepresentationState




@struct.dataclass
class LegoRearrangeRepresentationState(RepresentationState):
    curr_block: int
    blocks: chex.Array


     


class LegoRearrangeRepresentation(Representation):
    #pre_tile_action_dim: int
    #def __init__(self, tile_enum: Tiles,
    #             act_shape: Tuple[int, int]):
    def __init__(
            self,
            tile_enum: Tiles, 
            act_shape: Tuple[int, int], 
            env_shape: Tuple[int, int, int], 
            n_blocks: int,
            max_steps_multiple: int
            ):

        self.tile_enum = tile_enum
        #self.act_shape = act_shape
        self.env_shape = env_shape
        self.num_blocks = n_blocks

        
        self.max_steps = max_steps_multiple*self.num_blocks
        """
        self.blocks = jnp.zeros((self.num_blocks, 3), dtype="int32")
        for i in range(self.num_blocks):
            self.rng, subkey = jax.random.split(self.rng)
            pos = jax.random.randint(subkey, shape=(2,), minval = 0, maxval = self.env_map.shape[0], dtype=int)
            x = pos[0]
            z = pos[1]
            y = jnp.count_nonzero(self.env_map, 1)[x, z]

            self.blocks = self.blocks.at[i, 0].set(x)
            self.blocks = self.blocks.at[i, 1].set(y)
            self.blocks = self.blocks.at[i, 2].set(z)
            
            self.env_map = self.env_map.at[x,y,z].set(1)
        """

        self.moves = jnp.array([
            (0,0),
            (0,1),
            (0,-1),
            (1,0),
            (1,1),
            (1,-1),
            (-1,0),
            (-1,1),
            (-1,-1)   
        ])
        #print(jnp.count_nonzero(env_map, 1))
        

    def observation_shape(self):
        # Always observe static tile channel
        obs_shape = (2*(self.env_shape[0])-1, 2*(self.env_shape[1])-1, 2*(self.env_shape[2])-1)
        return (*obs_shape, len(self.tile_enum)+1)



    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        observation_shape = self.observation_shape()
        low = 0
        high = 1
        return spaces.Box(
            low, high, observation_shape, jnp.float32
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(9)
        #return spaces.Discrete((len(self.tile_enum)-1)
        #                       * math.prod(self.act_shape))
        

    def get_obs(self, rep_state: LegoRearrangeRepresentationState) -> chex.Array:
        blocks = rep_state.blocks
        curr_block = rep_state.curr_block

        curr_x, curr_y, curr_z = blocks[curr_block]
        x_offset = self.env_shape[0]-1-blocks[curr_block, 0]
        y_offset = self.env_shape[1]-1-blocks[curr_block, 1]
        z_offset = self.env_shape[2]-1-blocks[curr_block, 2]

        obs = jnp.zeros((2*self.env_shape[0]-1, 2*self.env_shape[1]-1, 2*self.env_shape[2]-1))
        for block in blocks:
            obs = obs.at[curr_x + x_offset, curr_y + y_offset, curr_z + z_offset].set(1)
        
        

        #obs = jax.lax.select(x_offset<0, obs.at[x_offset:,:,:].set(-1), obs.at[0:x_offset,:,:].set(-1))
        #obs = jax.lax.select(y_offset<0, obs.at[:,y_offset:,:].set(-1), obs.at[:,0:y_offset,:].set(-1))
        #obs = jax.lax.select(z_offset<0, obs.at[:,z_offset:,:].set(-1), obs.at[:,0:z_offset,:].set(-1))

        """
        for i in range(x_offset):      
            obs = obs.at[i,:,:].set(-1)
        for i in range(y_offset):
            obs = obs.at[:,i,:].set(-1)
        for i in range(z_offset):
            obs = obs.at[:,:,i].set(-1)

        for i in range(x_offset, 0):      
            obs = obs.at[i,:,:].set(-1)
        for i in range(y_offset, 0):
            obs = obs.at[:,i,:].set(-1)
        for i in range(z_offset, 0):
            obs = obs.at[:,:,i].set(-1)
        """

        return jax.nn.one_hot(obs, num_classes=len(self.tile_enum)+1, axis=-1)

    #@property
    #def per_tile_action_dim(self):
    #    return len(self.tile_enum) - 1
    
    def reset(self, rng: chex.PRNGKey):
        env_map = jnp.zeros(self.env_shape)
        blocks = jnp.zeros((self.num_blocks, 3), dtype="int32")
        for i in range(self.num_blocks):
            rng, subkey = jax.random.split(rng)
            pos = jax.random.randint(subkey, shape=(2,), minval = 0, maxval = self.env_shape[0], dtype=int)
            x = pos[0]
            z = pos[1]
            y = jnp.count_nonzero(env_map, 1)[x, z]

            blocks = blocks.at[i, 0].set(x)
            blocks = blocks.at[i, 1].set(y)
            blocks = blocks.at[i, 2].set(z)
            
            env_map = env_map.at[x,y,z].set(1)

        return LegoRearrangeRepresentationState(curr_block = 0, blocks = blocks)

    def step(self, env_map: chex.Array, action: chex.Array, rep_state: LegoRearrangeRepresentationState, step_idx: int):
        
        x_step, z_step = self.moves[action[0][0][0]]

        curr_y = rep_state.blocks[rep_state.curr_block, 1]
        curr_x = rep_state.blocks[rep_state.curr_block, 0]
        curr_z = rep_state.blocks[rep_state.curr_block, 2]

        #cond = (row[0] == curr_x) * (row[2] == curr_z) * (row[1] > curr_y)
        def subtract_if_condition_met(arr):
            def update_row(row):
                def subtract_one(row):
                    return row - jnp.array([0, 1, 0])

                def identity(row):
                    return row

                cond = (row[0] == curr_x) * (row[2] == curr_z) * (row[1] > curr_y)
                subtracted_row = jax.lax.cond(cond, subtract_one, identity, row)
                return subtracted_row

            updated_arr = jax.vmap(update_row)(arr)
            return updated_arr

        new_blocks = subtract_if_condition_met(rep_state.blocks)

        new_blocks = new_blocks.at[rep_state.curr_block, 0].add(x_step)
        new_blocks = new_blocks.at[rep_state.curr_block, 2].add(z_step)
        new_blocks = jnp.clip(new_blocks, a_min = 0, a_max = self.env_shape[0]-1)

        x = new_blocks[rep_state.curr_block, 0]
        z = new_blocks[rep_state.curr_block, 2]
     
        new_height = jnp.count_nonzero(env_map, 1)[x, z]
 
        max_height = self.get_max_height
        new_blocks = new_blocks.at[rep_state.curr_block, 1].set(jnp.count_nonzero(env_map, 1)[x, z])        

        return_blocks = jax.lax.select(new_height > max_height, rep_state.blocks, new_blocks)
        return_blocks = jax.lax.select(((x == curr_x) & (z == curr_z)), rep_state.blocks, new_blocks)

        return_map = self.get_env_map(return_blocks)
        return_state = LegoRearrangeRepresentationState(curr_block = (rep_state.curr_block + 1)%self.num_blocks, blocks = return_blocks)

        return return_map, None, return_state
    
    @property
    def get_max_height(self):
        return self.env_shape[1]-1

    def get_env_map(self, blocks):
        env_map = jnp.zeros(self.env_shape)
        for block in blocks:
            env_map = env_map.at[block[0], block[1], block[2]].set(1)
        
        return env_map