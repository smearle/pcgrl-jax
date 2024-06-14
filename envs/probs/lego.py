from enum import IntEnum
from typing import Optional

import chex
from flax import struct
import jax
from jax.experimental import checkify
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.utils import Tiles
from .problem import Problem

class LegoTiles(IntEnum):
    EMPTY = 0
    N3005 = 1 #1x1 piece, number 3005
    N3004 = 2 #2x1 piece, number 3005
    N3031 = 3 #4x4 plate, number 3031

tileNames = ["empty", "3005", "3004", "3031"]
tileDims = jnp.array([(0,0,0), (1,3,1), (1,3,2), (4,1,4)])

@struct.dataclass
class LegoProblemState:
    stats: Optional[chex.Array] = None
    ctrl_trgs: Optional[chex.Array] = None
    done: Optional[bool] = False
    

class LegoMetrics(IntEnum):
    AVG_HEIGHT = 0
    FOOTPRINT = 7 #num blocks touching the floor
    AVG_EUCLIDEAN = 2
    CENTER = 3
    HOUSE = 6
    TABLE = 1
    COVERED_VOL = 4
    STAIRS = 5

class LegoProblem(Problem):
    #tile_size = np.int8(16)
    stat_weights = jnp.zeros((len(LegoMetrics)))
    metrics_enum = LegoMetrics

    def __init__(self, map_shape, ctrl_metrics, n_blocks, reward):
        self.tile_enum = LegoTiles
        # self.map_shape = map_shape
        self.metrics_enum = LegoMetrics
        self.n_blocks = n_blocks
        self.metric_bounds = self.get_metric_bounds(map_shape)

        stat_trgs = np.zeros(len(LegoMetrics))
        stat_trgs[LegoMetrics.AVG_HEIGHT] = sum([i for i in range(self.n_blocks)])/self.n_blocks
        stat_trgs[LegoMetrics.FOOTPRINT] = 1
        stat_trgs[LegoMetrics.AVG_EUCLIDEAN] = 0
        stat_trgs[LegoMetrics.CENTER] = 0
        stat_trgs[LegoMetrics.HOUSE] = 1
        stat_trgs[LegoMetrics.TABLE] = 1.0
        stat_trgs[LegoMetrics.COVERED_VOL] = 3*(self.n_blocks-1)*(4*4-1)
        stat_trgs[LegoMetrics.STAIRS] = n_blocks-1
        self.stat_trgs = jnp.array(stat_trgs)

        self.stat_weights = jnp.where(jnp.isin(jnp.arange(self.stat_weights.size), jnp.array(reward)), 1, self.stat_weights)
        
        super().__init__(map_shape = map_shape, ctrl_metrics = ctrl_metrics)

    def gen_init_map(self, rng):
        return jnp.zeros(self.map_shape)
        #return gen_init_map(rng, self.tile_enum, self.map_shape,
        #                       self.tile_probs)

    def get_metric_bounds(self, map_shape):
        bounds = [None] * len(LegoMetrics)
        bounds[LegoMetrics.AVG_HEIGHT] = [0, map_shape[1]]
        bounds[LegoMetrics.FOOTPRINT] = [1, map_shape[0] * map_shape[2]]
        bounds[LegoMetrics.AVG_EUCLIDEAN] = [1, (map_shape[0]**2+map_shape[2]**2)**(0.5)]
        bounds[LegoMetrics.HOUSE] = [0, 1]
        bounds[LegoMetrics.TABLE] = [0, 1]
        bounds[LegoMetrics.STAIRS] = [0, self.n_blocks-1]
        bounds[LegoMetrics.COVERED_VOL] = [0, 3*(self.n_blocks*-1)*(4*4-1)]

        cntr_x, cntr_z = (map_shape[0]-1)//2, (map_shape[2]-1)//2        

        bounds[LegoMetrics.CENTER] = [0, ((cntr_x+1)**2+(cntr_z+1)**2)**(0.5)]
        return np.array(bounds)

    def get_curr_stats(self, blocks: chex.Array, env_map: chex.Array):
        """Get relevant metrics from the current state of the environment."""
        
        total_heights = jnp.sum(blocks[:,1], axis = None)
        avg_height = total_heights/blocks.shape[0]
        #max_height = jnp.max(blocks[:,1])

        footprint = jnp.count_nonzero(jnp.where(blocks[:,1] == 0, 1, 0))

        cntr_x, cntr_z = (env_map.shape[0]-1)//2, (env_map.shape[2]-1)//2
        
        # Get average distance of blocks to center
        xy_diffs_cntr = jnp.abs(blocks[:,(0,2)] - jnp.array([cntr_x, cntr_z]))
        cntr_dist = jnp.sum(jnp.linalg.norm(xy_diffs_cntr, axis=1))
        avg_cntr_dist = cntr_dist/blocks.shape[0]

        # Get average euclidean distance between (unique) blocks
        xy_diffs_pairwise = jnp.abs(blocks[:,(0,2)][:,None] - blocks[:,(0,2)])
        pair_dist = jnp.linalg.norm(xy_diffs_pairwise, axis=2)
        avg_euclidean = jnp.sum(pair_dist) / (blocks.shape[0] * (blocks.shape[0] - 1))

        stats = jnp.zeros(len(LegoMetrics))
        stats = stats.at[LegoMetrics.AVG_HEIGHT].set(avg_height)
        stats = stats.at[LegoMetrics.FOOTPRINT].set(footprint)
        stats = stats.at[LegoMetrics.AVG_EUCLIDEAN].set(avg_euclidean)
        stats = stats.at[LegoMetrics.CENTER].set(avg_cntr_dist)

        #table_ness
        roof_x = blocks[-1,0]
        roof_z = blocks[-1,2]
        
        table = 0.0
        for block in blocks:
            curr_x, curr_y, curr_z, curr_block_type = block
            is_in_table_position = jnp.logical_and(
                jnp.logical_and(
                    jnp.logical_or(
                        curr_x == roof_x,
                        curr_x == roof_x + 3
                    ),
                    jnp.logical_or(
                        curr_z == roof_z,
                        curr_z == roof_z + 3
                    )
                ),
                jnp.logical_or(    
                    jnp.logical_and(
                        curr_block_type != 3,
                        curr_y < 3*(blocks.shape[0]//4)
                    ),
                    jnp.logical_and(
                        curr_block_type == 3,
                        curr_y == 3*(blocks.shape[0]//4)
                    )
                    
                    ) 
                )  
            #is_in_table_position = True
            #jax.debug.print("table position: {t}. before: {table}. after: {ta}", table=table, t=is_in_table_position, ta=table + is_in_table_position)
            table += is_in_table_position
        
        stats = stats.at[LegoMetrics.TABLE].set(table/float(blocks.shape[0]))

        #COVERED_VOL
        zero_mask = env_map == 0
        def scan_fun(carry, x):
            nonzero_seen = carry | (x != 0)
            return nonzero_seen, nonzero_seen
        # Transpose the array to move the second axis to the front
        transposed_env_map = jnp.transpose(env_map, (1, 0, 2))

        # Initialize the carry with False
        init = jnp.zeros_like(transposed_env_map[0, :, :], dtype=bool)

        # Perform the scan in reverse order along the second axis (which is now the leading axis)
        _, transposed_nonzero_mask = jax.lax.scan(scan_fun, init, jnp.flip(transposed_env_map, axis=0))

        # Transpose the mask back to the original order
        nonzero_mask = jnp.transpose(jnp.flip(transposed_nonzero_mask, axis=0), (1, 0, 2))

        # Use the masks to count the number of elements that are zero and have a nonzero element at a higher index along the second axis
        covered_vol = jnp.sum(zero_mask & nonzero_mask)
        
        stats = stats.at[LegoMetrics.COVERED_VOL].set(covered_vol)


        #STAIRS
        nonzero_mask = env_map != 0
        zero_mask_shifted = jnp.pad(env_map==0, ((0, 0), (0, 1), (0, 0)))[:,1:,:] == 0
        
        #Step 3: Create a mask of nonzero elements adjacent to the nonzero elements at the same index in axis b
        # Check for adjacent elements along axis a
        nonzero_mask_adjacent_a = jnp.pad(nonzero_mask, ((1, 0), (0, 0), (0, 0)))
        nonzero_mask_adjacent_a = jnp.logical_or(nonzero_mask_adjacent_a[:-1], nonzero_mask_adjacent_a[1:])

        # Check for adjacent elements along axis c
        nonzero_mask_adjacent_c = jnp.pad(nonzero_mask, ((0, 0), (0, 0), (1, 0)))
        nonzero_mask_adjacent_c = jnp.logical_or(nonzero_mask_adjacent_c[..., :-1], nonzero_mask_adjacent_c[..., 1:])

        # Combine the masks along axis a and c
        nonzero_mask_adjacent = jnp.logical_or(nonzero_mask_adjacent_a, nonzero_mask_adjacent_c)

        # Step 4: Combine the masks using logical AND operation
        combined_mask = jnp.logical_and(nonzero_mask[:-1, :-1, :-1], zero_mask_shifted[:-1, :-1, :-1])
        combined_mask = jnp.logical_and(combined_mask, nonzero_mask_adjacent[:-1, :-1, :-1])

        #indices = jnp.arange(combined_mask.shape[1]).reshape(1, combined_mask.shape[1], 1)

        #stairs = jnp.sum(combined_mask*indices)
        stairs = jnp.sum(combined_mask)

        stats = stats.at[LegoMetrics.STAIRS].set(stairs)        
        
        done = jnp.sum(jnp.where(stats == self.stat_trgs, 0, 1) * self.stat_weights) == 0
        
        state = LegoProblemState(
            stats=stats, 
            ctrl_trgs=self.get_stat_trgs,
            done = done)
        
        
        return state
    





    #########################################
    def queue_ctrl_trgs(self, queued_state, ctrl_trgs):
        queued_state = queued_state.replace(queued_ctrl_trgs=ctrl_trgs, has_queued_ctrl_trgs=True)
        return queued_state
        #raise NotImplementedError

    @property
    def get_stat_trgs(self):
        return self.stat_trgs
   
    def reset(self, blocks: chex.Array, env_map: chex.Array):
        old_stats = self.get_curr_stats(blocks, env_map).stats
        
        state = LegoProblemState(ctrl_trgs = self.get_stat_trgs, stats = old_stats, done = False)
        reward = self.get_reward(state.stats, old_stats, self.stat_weights, self.stat_trgs, self.ctrl_threshes)
        return reward, state

    

    def step(self, env_map: chex.Array, state: LegoProblemState, blocks:chex.Array):
        new_state = self.get_curr_stats(blocks=blocks, env_map = env_map)
        reward = self.get_reward(new_state.stats, state.stats, self.stat_weights, state.ctrl_trgs, self.ctrl_threshes)
        new_state = new_state.replace(
            ctrl_trgs=state.ctrl_trgs,
        )
        return reward, new_state

    
    def get_reward(self, stats, old_stats, stat_weights, stat_trgs, ctrl_threshes):
        """
        ctrl_threshes: A vector of thresholds for each metric. If the metric is within
            an interval of this size centered at its target value, it has 0 loss.
        """
 
        prev_loss = jnp.abs(stat_trgs - old_stats)
        
        prev_loss = jnp.clip(prev_loss - ctrl_threshes, 0)

        loss = jnp.abs(stat_trgs - stats)

        loss = jnp.clip(loss - ctrl_threshes, 0)

        reward = prev_loss - loss

        reward = jnp.where(stat_trgs == jnp.inf, stats - old_stats, reward)

        reward = jnp.where(stat_trgs == -jnp.inf, old_stats - stats, reward)

        reward *= stat_weights

        reward = jnp.sum(reward)
        
        return reward
    
    