from abc import ABC
import math
from typing import Tuple
import copy
import os

import chex
from flax import struct
from gymnax.environments import spaces
import jax
import jax.numpy as jnp

from envs.utils import Tiles, pad_env_map
from .representation import Representation, RepresentationState
from ..probs.lego import tileNames, tileDims, LegoMetrics

block_masks = jnp.zeros((4,6,3,6))
for i in range(4):
    dims = tileDims[i]
    block_masks = block_masks.at[i, 0:dims[0], 0:dims[1], 0:dims[2]].set(i)


@struct.dataclass
class LegoRearrangeRepresentationState(RepresentationState):
    curr_block: int
    blocks: chex.Array
    last_action: int=0

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
            max_steps_multiple: int,
            reward: Tuple[str]
            ):

        self.tile_enum = tile_enum
        #self.act_shape = act_shape
        self.env_shape = env_shape
        self.num_blocks = n_blocks
        self.reward = reward
        
        self.max_steps = max_steps_multiple*self.num_blocks

        self.moves = jnp.array([
            (0,0), #no move, goes to top
            (0,1),
            (0,-1),
            (1,0),
            (-1,0),
            (0,0) # no move, does not go to top  
        ])
        #print(jnp.count_nonzero(env_map, 1))
        

    def observation_shape(self):
        # Always observe static tile channel
        obs_shape = (2*(self.env_shape[0])+1, 2*(self.env_shape[1])+1, 2*(self.env_shape[2])+1)
        return (*obs_shape, len(self.tile_enum)+1)



    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        observation_shape = self.observation_shape()
        low = 0
        high = 2        
        return spaces.Box(
            low, high, observation_shape, jnp.float32
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.moves.shape[0])
        #return spaces.Discrete((len(self.tile_enum)-1)
        #                       * math.prod(self.act_shape))
    


    def get_obs(self, rep_state: LegoRearrangeRepresentationState) -> chex.Array:
        blocks = rep_state.blocks
        curr_block = rep_state.curr_block

        
        blockdims = tileDims[blocks[:,3]]

        obs_shape_x, obs_shape_y, obs_shape_z = self.observation_shape()[:3]

        #curr_x, curr_y, curr_z = blocks[curr_block]
        x_offset = obs_shape_x//2-blocks[curr_block, 0]
        y_offset = obs_shape_y//2-blocks[curr_block, 1]
        z_offset = obs_shape_z//2-blocks[curr_block, 2]

        obs = jnp.zeros((obs_shape_x, obs_shape_y, obs_shape_z))

        obs = pad_env_map(obs, block_masks[0].shape)

        #set obs
        # obs = obs.at[x_offset: x_offset+self.env_shape[0], y_offset: y_offset+self.env_shape[1], z_offset: z_offset+self.env_shape[2]].set(1)
        obs = jax.lax.dynamic_update_slice(
            obs,
            jnp.ones(self.env_shape),
            (x_offset, y_offset, z_offset),
        )
        
        #obs = obs.at[blocks[:, 0]+x_offset:blocks[:,0]+x_offset+blockdims[:,0], blocks[:, 1]+y_offset:blocks[:,1]+y_offset+blockdims[:,1], blocks[:, 2]+z_offset:blocks[:,2]+z_offset+blockdims[:,2]].set(blocks[:, 3])
        #obs = obs.at[blocks[:, 0]+x_offset, blocks[:, 1]+y_offset, blocks[:, 2]+z_offset+blocks[:, 3]-1].set(blocks[:, 3])
        
        for i in range(blocks.shape[0]):
            x = blocks[i, 0] + x_offset
            #x_stop = x_start + blockdims[i, 0]
            y = blocks[i, 1] + y_offset
            #y_stop = y_start + blockdims[i, 1]
            z = blocks[i, 2] + z_offset
            #z_stop = z_start + blockdims[i, 2]

            
            mask = block_masks[blocks[i,3]]

            # First, create a slice of env_map with the same shape as mask
            obs_slice = jax.lax.dynamic_slice(obs, (x, y, z), mask.shape)

            # Apply the mask to the slice
            updated_slice = jnp.where(mask != 0, mask, obs_slice)

            # Update the slice of env_map at the position (x, y, z) with the masked slice
            obs = jax.lax.dynamic_update_slice(obs, updated_slice, (x, y, z))
            
        obs = obs[:obs_shape_x, :obs_shape_y, :obs_shape_z]
        obs = jax.nn.one_hot(obs, num_classes=len(self.tile_enum)+1, axis=-1)
        return obs

    #@property
    #def per_tile_action_dim(self):
    #    return len(self.tile_enum) - 1

    @property
    def is_house_or_table_builder(self):
        return LegoMetrics.HOUSE in self.reward or LegoMetrics.TABLE in self.reward
    

    def reset(self, rng: chex.PRNGKey):
        env_map = jnp.zeros(self.env_shape)
        env_map = pad_env_map(env_map, block_masks[0].shape)

        blocks = jnp.zeros((self.num_blocks, 4), dtype="int32")

        rng, subkey = jax.random.split(rng)

        #blocks of size 1 along z dim
        zpos1 = jax.random.randint(subkey, shape=(self.num_blocks//2,), minval = 0, maxval = self.env_shape[0], dtype=int) 
        #blocks of size 2 along z dim
        rng, subkey = jax.random.split(rng)
        zpos2 = jax.random.randint(subkey, shape=(self.num_blocks-self.num_blocks//2-1,), minval = 0, maxval = self.env_shape[0]-1, dtype=int)
        #blocks of size 6 along z dim
        rng, subkey = jax.random.split(rng)
        zpos3 = jax.random.randint(subkey, shape=(1,), minval = 0, maxval = self.env_shape[0]-4, dtype=int)

        rng, subkey = jax.random.split(rng)
        xpos1 = jax.random.randint(subkey, shape=(self.num_blocks-1,), minval = 0, maxval = self.env_shape[0], dtype=int) 
        rng, subkey = jax.random.split(rng)
        xpos2 = jax.random.randint(subkey, shape=(1,), minval = 0, maxval = self.env_shape[2]-4, dtype=int)

        z = jnp.concatenate([zpos1, zpos2, zpos3], axis=0)
        x = jnp.concatenate([xpos1, xpos2], axis=0)

         

        def stack_blocks(carry, i):
            blocks, env_map, x, z = carry
            blocktype = 1 + (i>=self.num_blocks//2)
            y = self.get_max_height - tileDims[blocktype][1]

            blocks = blocks.at[i, 0].set(x[i])
            blocks = blocks.at[i, 1].set(y)
            blocks = blocks.at[i, 2].set(z[i])
            blocks = blocks.at[i, 3].set(blocktype)

            mask = block_masks[blocktype]
            mask = jnp.clip(mask, 0, 1)
            
            # First, create a slice of env_map with the same shape as mask
            env_slice = jax.lax.dynamic_slice(env_map, (x[i], y, z[i]), mask.shape)

            # Apply the mask to the slice
            updated_slice = jnp.where(mask != 0, mask, env_slice)

            # Update the slice of env_map at the position (x, y, z) with the masked slice
            env_map = jax.lax.dynamic_update_slice(env_map, updated_slice, (x[i], y, z[i]))
            
            return (blocks, env_map, x, z), 0
        
        carry, _ = jax.lax.scan(stack_blocks, (blocks, env_map, x, z), jnp.arange(self.num_blocks)-1)

        blocks, env_map, _, _ = carry

        #if we are making a house or table, keep things simple for now
        blocks = jax.lax.select(self.is_house_or_table_builder, blocks.at[:, 3].set(1), blocks)
        """
        #if we are making a house or table, num blocks is variable
        rng, subkey = jax.random.split(rng)
        leg_ht = jax.random.randint(subkey, shape=(1,), minval = 1, maxval = 6, dtype=int)

        max_leg_blocks = 5*5 #for maxval=6
        indices = jnp.arange(max_leg_blocks)
        leg_blocks = 4*jnp.squeeze(leg_ht)
        
        slice = jnp.zeros((1, 4), dtype="int32")
        # Reshape indices < leg_blocks to have the same shape as slice and blocks
        condition = jnp.reshape(indices >= leg_blocks, (max_leg_blocks, 1))

        # Use jnp.where to select the appropriate elements from slice or blocks
        table_blocks = jnp.where(condition, slice, blocks)

        # Then use the select function as before
        blocks = jax.lax.select(self.is_house_or_table_builder, table_blocks, blocks)
        """
        #add one large flat block 
        blocks = blocks.at[self.num_blocks-1, 3].set(3)#TEST
        mask = block_masks[3]
        env_slice = jax.lax.dynamic_slice(env_map, (x[-1], self.get_max_height-1, z[-1]), mask.shape)
        updated_slice = jnp.where(mask != 0, mask, env_slice)
        env_map = jax.lax.dynamic_update_slice(env_map, updated_slice, (x[-1], self.get_max_height-1, z[-1]))


        def block_fall_fn(block):
            temp_map = pad_env_map(self.get_env_map(blocks), block_masks[0].shape)
            def cond_fn(state):
                block, temp_map = state
                mask = jnp.clip(block_masks[block[3]], 0, 1)

                flat_mask = jnp.expand_dims(mask[:,0,:], 1)
                slice_shape = flat_mask.shape
                slice_start_indices = (block[0], block[1]-1, block[2])
                temp_slice = jax.lax.dynamic_slice(temp_map, slice_start_indices, slice_shape)
                temp_slice_zeroed = jnp.where(temp_slice != 0, 1, 0)
                return jnp.logical_and(
                    block[1] > 0, 
                    jnp.all(temp_slice_zeroed + flat_mask <= 1)
                )
  
            def body_fn(state):
                block, temp_map = state
                #jax.debug.breakpoint()

                mask = jnp.clip(block_masks[block[3]], 0, 1)

                env_slice = jax.lax.dynamic_slice(temp_map, (block[0], block[1], block[2]), mask.shape)
                updated_slice = jnp.where(mask != 0, 0, env_slice)
                temp_map = jax.lax.dynamic_update_slice(temp_map, updated_slice, (block[0], block[1], block[2]))
                
                block = block.at[1].set(block[1] - 1)

                env_slice = jax.lax.dynamic_slice(temp_map, (block[0], block[1], block[2]), mask.shape)
                updated_slice = jnp.where(mask != 0, mask, env_slice)
                temp_map = jax.lax.dynamic_update_slice(temp_map, updated_slice, (block[0], block[1], block[2]))
                #jax.debug.breakpoint()
                return block, temp_map

            block, temp_map = jax.lax.while_loop(cond_fn, body_fn, (block, temp_map))

            return block

        for rownum in range(blocks.shape[0]):
            block = block_fall_fn(blocks[rownum])
            blocks = blocks.at[rownum].set(block)

        return LegoRearrangeRepresentationState(curr_block = 0, blocks = blocks, last_action = 0)

    def step(self, env_map: chex.Array, action: chex.Array, rep_state: LegoRearrangeRepresentationState, step_idx: int, rng):
        action_ind = action[0][0][0]
        x_step, z_step = self.moves[action_ind]

        #move block
        new_blocks = rep_state.blocks.at[rep_state.curr_block, 0].add(x_step)
        new_blocks = new_blocks.at[rep_state.curr_block, 2].add(z_step)

        #set height to max to start with. will fall to top of existing blocks, or cancel move if overlap
        max_height = self.get_max_height - tileDims[new_blocks[rep_state.curr_block, 3], 1]
        new_blocks = new_blocks.at[rep_state.curr_block,1].set(max_height)

        #all blocks fall if empty space below them           
        ordered_inds = jnp.argsort(new_blocks[:,1])
        ordered_blocks = new_blocks[ordered_inds]
       
        def block_fall_fn(block):
            temp_map = pad_env_map(self.get_env_map(ordered_blocks), block_masks[0].shape)
            def cond_fn(state):
                block, temp_map = state
                mask = block_masks[block[3]]
                mask = jnp.clip(mask, 0, 1)
                flat_mask = jnp.expand_dims(mask[:,0,:], 1)
                slice_shape = flat_mask.shape
                slice_start_indices = (block[0], block[1]-1, block[2])
                temp_slice = jax.lax.dynamic_slice(temp_map, slice_start_indices, slice_shape)
                temp_slice_zeroed = jnp.where(temp_slice != 0, 1, 0)
                return jnp.logical_and(
                    block[1] > 0, 
                    jnp.all(temp_slice_zeroed + flat_mask <= 1)
                )
  
            def body_fn(state):
                block, temp_map = state
                mask = block_masks[block[3]]
                env_slice = jax.lax.dynamic_slice(temp_map, (block[0], block[1], block[2]), mask.shape)
                updated_slice = jnp.where(mask != 0, 0, env_slice)
                temp_map = jax.lax.dynamic_update_slice(temp_map, updated_slice, (block[0], block[1], block[2]))
                
                block = block.at[1].set(block[1] - 1)

                env_slice = jax.lax.dynamic_slice(temp_map, (block[0], block[1], block[2]), mask.shape)
                updated_slice = jnp.where(mask != 0, mask, env_slice)
                temp_map = jax.lax.dynamic_update_slice(temp_map, updated_slice, (block[0], block[1], block[2]))
                               
                return block, temp_map

            block, temp_map = jax.lax.while_loop(cond_fn, body_fn, (block, temp_map))

            return block

        for rownum in range(ordered_blocks.shape[0]):
            block = block_fall_fn(ordered_blocks[rownum])
            ordered_blocks = ordered_blocks.at[rownum].set(block)

        new_blocks = ordered_blocks[jnp.argsort(ordered_inds)]
            
        curr_block = new_blocks[rep_state.curr_block].at[1].set(max_height+1)
        curr_block = block_fall_fn(curr_block)
        

        #check for out of bounds
        new_blocks = jax.lax.select(new_blocks[rep_state.curr_block, 2] >=self.env_shape[2] - tileDims[new_blocks[rep_state.curr_block, 3], 2] - 1, rep_state.blocks, new_blocks)
        new_blocks = jax.lax.select(new_blocks[rep_state.curr_block, 2] < 0, rep_state.blocks, new_blocks)
        new_blocks = jax.lax.select(new_blocks[rep_state.curr_block, 0] >= self.env_shape[0] - tileDims[new_blocks[rep_state.curr_block, 3], 0] - 1, rep_state.blocks, new_blocks)
        new_blocks = jax.lax.select(new_blocks[rep_state.curr_block, 0] < 0, rep_state.blocks, new_blocks)

        new_height = new_blocks[rep_state.curr_block,1]

        return_blocks = jax.lax.select(new_height > max_height, rep_state.blocks, new_blocks)
        return_blocks = jax.lax.select(action_ind == 5, rep_state.blocks, return_blocks)


        return_map = self.get_env_map(return_blocks)
        
        rng, subkey = jax.random.split(rng)

        next_block = (rep_state.curr_block + 1)%self.num_blocks

        """
        def cond_fun(carry):
            next_block,_ = carry
            return return_blocks[next_block,3] != 0
        def body_fun(carry):
            next_block,_ = carry
            return (next_block + 1)%self.num_blocks, 0
        
        next_block = jax.lax.while_loop(cond_fun, body_fun, (next_block, 0))
        """

        return_state = LegoRearrangeRepresentationState(
            curr_block = next_block,
            blocks = return_blocks,
            last_action=action[0][0][0]
            )

        return return_map, rng, return_state
    
    @property
    def get_max_height(self):
        return self.env_shape[1]

    def get_env_map(self, blocks):
        #adding extra space to env map to allow for use of masks
        env_map = jnp.zeros((self.env_shape[0] + block_masks.shape[1], self.env_shape[1] + block_masks.shape[2], self.env_shape[2] + block_masks.shape[3
                                                                                                                                                       ]))
        #env_map = jnp.zeros(self.env_shape)
        
        for block in blocks:
            mask = block_masks[block[3]]
            mask = jnp.clip(mask, 0, 1)
        #    jax.debug.breakpoint()
            env_slice = jax.lax.dynamic_slice(env_map, (block[0], block[1], block[2]), mask.shape)
            updated_slice = jnp.where(mask != 0, mask, env_slice)
            env_map = jax.lax.dynamic_update_slice(env_map, updated_slice, (block[0], block[1], block[2]))
        #    jax.debug.breakpoint()
        
        #jax.debug.breakpoint()
        return env_map[:self.env_shape[0], :self.env_shape[1], :self.env_shape[2]]