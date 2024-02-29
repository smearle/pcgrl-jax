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

from envs.utils import Tiles
from .representation import Representation, RepresentationState

def render_mpds(blocks, i, savedir, config):

                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                
                for num in range(blocks.shape[0]):
                    curr_blocks = blocks[num,:,:]
                    rotation = blocks[num]
                    curr_block = blocks[num]

                    savename = os.path.join(savedir, f"{num}.mpd")
                    
                    f = open(savename, "a")
                    f.write("0/n")
                    f.write("0 Name: New Model.ldr\n")
                    f.write("0 Author:\n")
                    f.write("\n")

                    for x in range(config.map_width):
                        for z in range(config.map_width):
                            lego_block_name = "3005"
                            block_color = "2 "
                            if curr_block == num:
                                block_color = "7 "
                            
                            y_offset = -3#-24

                            x_lego = x * 20  + config.map_width - 1
                            y_lego =  0#(1)*(LegoDimsDict[lego_block_name][1])
                            z_lego = z * 20 + config.map_width - 1

                            #print(block.x, block.y, block.z)
                            
                            f.write("1 ")
                            f.write(block_color)
                            f.write(str(x_lego) + ' ' + str(y_lego) + ' ' + str(z_lego) + ' ')
                            f.write("1 0 0 0 1 0 0 0 1 ")
                            f.write(lego_block_name + ".dat")
                            f.write("\n")
                    
                    y_offset = -24
                    for b in range(curr_blocks.shape[0]):
                        lego_block_name = "3005"
                        block_color = "7 "
                        x_lego = curr_blocks[b, 0] * 20  + config.map_width - 1
                        y_lego = curr_blocks[b, 1] * (-24) + y_offset
                        z_lego = curr_blocks[b, 2] * 20 + config.map_width - 1

                        f.write("1 ")
                        f.write(block_color)
                        f.write(str(x_lego) + ' ' + str(y_lego) + ' ' + str(z_lego) + ' ')
                        f.write("1 0 0 0 1 0 0 0 1 ")
                        f.write(lego_block_name + ".dat")
                        f.write("\n")
                    f.close()


@struct.dataclass
class LegoRearrangeRepresentationState(RepresentationState):
    curr_block: int
    blocks: chex.Array
    rotation: int=0
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
            max_steps_multiple: int
            ):

        self.tile_enum = tile_enum
        #self.act_shape = act_shape
        self.env_shape = env_shape
        self.num_blocks = n_blocks

        
        self.max_steps = max_steps_multiple*self.num_blocks

        self.moves = jnp.array([
            (0,0),
            (0,1),
            (0,-1),
            (1,0),
            #(1,1),
            #(1,-1),
            (-1,0),
            #(-1,1),
            #(-1,-1)   
        ])
        #print(jnp.count_nonzero(env_map, 1))
        

    def observation_shape(self):
        # Always observe static tile channel
        obs_shape = (2*(self.env_shape[0])+1, 2*(self.env_shape[1])+1, 2*(self.env_shape[2])+1)
        return (*obs_shape, len(self.tile_enum)+2)



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
    
    def flip_0(self, x: int, z: int) -> (int, int):
        return 0-x, z 
    
    def flip_1(self, x: int, z: int) -> (int, int):
        return x, 0-z
    
    def rot90_1(self, x: int, z: int) -> (int, int):
        return 0-z, x
    
    def rot90_2(self, x: int, z: int) -> (int, int):
        return 0-x, 0-z
    
    def rot90_3(self, x: int, z: int) -> (int, int):
        return z, 0-x

    def identity_perturb(self, x: int, z: int):
        return x, z

    def unperturb_action(self, x: int, z: int, rotation:int) -> int:
        new_x, new_z = x, z
        new_x, new_z = jax.lax.cond(rotation == 1, self.flip_0, self.identity_perturb, x, z)
        new_x, new_z = jax.lax.cond(rotation == 2, self.flip_1, self.identity_perturb, x, z)
        new_x, new_z = jax.lax.cond(rotation == 3, self.rot90_1, self.identity_perturb, x, z)
        new_x, new_z = jax.lax.cond(rotation == 4, self.rot90_2, self.identity_perturb, x, z)
        new_x, new_z = jax.lax.cond(rotation == 5, self.rot90_3, self.identity_perturb, x, z)

        return new_x, new_z

    def perturb_obs(self, rotation: int, obs: chex.Array) -> chex.Array:
        perturbed_obs = jax.lax.select(rotation==0, obs, obs)
        perturbed_obs = jax.lax.select(rotation == 1, jnp.flip(obs, 0), obs) # flip x axis
        perturbed_obs = jax.lax.select(rotation == 2, jnp.flip(obs, 2), obs) # flip z axis
        perturbed_obs = jax.lax.select(rotation == 3, jnp.rot90(obs, 1, (0,2)), obs) # rotate 90
        perturbed_obs = jax.lax.select(rotation == 4, jnp.rot90(obs, 2, (0,2)), obs) # rotate 180
        perturbed_obs = jax.lax.select(rotation == 5, jnp.rot90(obs, 3, (0,2)), obs) # rotate 270
        return perturbed_obs




    def get_obs(self, rep_state: LegoRearrangeRepresentationState) -> chex.Array:
        blocks = rep_state.blocks
        curr_block = rep_state.curr_block
        rotation = rep_state.rotation

        obs_shape_x, obs_shape_y, obs_shape_z = self.observation_shape()[:3]

        #curr_x, curr_y, curr_z = blocks[curr_block]
        x_offset = obs_shape_x//2-blocks[curr_block, 0]
        y_offset = obs_shape_y//2-blocks[curr_block, 1]
        z_offset = obs_shape_z//2-blocks[curr_block, 2]

        obs = jnp.zeros((obs_shape_x, obs_shape_y, obs_shape_z))

        #set obs
        # obs = obs.at[x_offset: x_offset+self.env_shape[0], y_offset: y_offset+self.env_shape[1], z_offset: z_offset+self.env_shape[2]].set(1)
        obs = jax.lax.dynamic_update_slice(
            obs,
            jnp.ones(self.env_shape),
            (x_offset, y_offset, z_offset),
        )
        
        obs = obs.at[blocks[:, 0]+x_offset, blocks[:, 1]+y_offset, blocks[:, 2]+z_offset].set(2)

        #mark current block
        obs = obs.at[blocks[curr_block,0]+x_offset, blocks[curr_block,1]+y_offset,blocks[curr_block,2]+z_offset].set(3)
        #mark center block
        #obs = obs.at[(self.env_shape[0]-1)//2+x_offset, (self.env_shape[1]-1)//2+y_offset, (self.env_shape[2]-1)//2+z_offset].set(4)
        
        obs = jax.nn.one_hot(obs, num_classes=len(self.tile_enum)+2, axis=-1)
        return obs

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
        
        rotation = 0#jax.random.randint(subkey,shape=(1,), minval =0, maxval=3, dtype=int)[0]

        return LegoRearrangeRepresentationState(curr_block = 0, blocks = blocks, rotation=rotation, last_action = 0)

    def step(self, env_map: chex.Array, action: chex.Array, rep_state: LegoRearrangeRepresentationState, step_idx: int, rng):
        
        rotation = rep_state.rotation

        #rotation = 4

        x_step, z_step = self.moves[action[0][0][0]]

        #x_step, z_step = -1, 0
        x_step, z_step = self.unperturb_action(x_step, z_step, rotation)
        
        curr_y = rep_state.blocks[rep_state.curr_block, 1]
        curr_x = rep_state.blocks[rep_state.curr_block, 0]
        curr_z = rep_state.blocks[rep_state.curr_block, 2]

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


        new_blocks_x = jnp.clip(new_blocks[:,0], a_min = 0, a_max = self.env_shape[0]-1)
        new_blocks_z = jnp.clip(new_blocks[:,2], a_min = 0, a_max = self.env_shape[2]-1)
        new_blocks = new_blocks.at[:,0].set(new_blocks_x)
        new_blocks = new_blocks.at[:,2].set(new_blocks_z)
       
        x = new_blocks[rep_state.curr_block, 0]
        z = new_blocks[rep_state.curr_block, 2]
     
        new_height = jnp.count_nonzero(env_map, 1)[x, z]
 
        max_height = self.get_max_height
        new_blocks = new_blocks.at[rep_state.curr_block, 1].set(jnp.count_nonzero(env_map, 1)[x, z])        

        return_blocks = jax.lax.select(new_height > max_height, rep_state.blocks, new_blocks)

        

        return_blocks = jax.lax.select(((x == curr_x) & (z == curr_z)), rep_state.blocks, new_blocks)

        return_map = self.get_env_map(return_blocks)
        
        rng, subkey = jax.random.split(rng)
        rotation = 0#ax.random.randint(subkey,shape=(1,), minval =0, maxval=3, dtype=int)[0]

        return_state = LegoRearrangeRepresentationState(
            curr_block = (rep_state.curr_block + 1)%self.num_blocks, 
            blocks = return_blocks,
            rotation = rotation,
            last_action=action[0][0][0]
            )

        return return_map, rng, return_state
    
    @property
    def get_max_height(self):
        return self.env_shape[1]-1

    def get_env_map(self, blocks):
        env_map = jnp.zeros(self.env_shape)
        for block in blocks:
            env_map = env_map.at[block[0], block[1], block[2]].set(1)
        
        return env_map