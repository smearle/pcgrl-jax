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
    FOOTPRINT = 5 #num blocks touching the floor
    AVG_EUCLIDEAN = 2
    CENTER = 3
    HOUSE = 4
    TABLE = 1

class LegoProblem(Problem):
    #tile_size = np.int8(16)
    stat_weights = jnp.zeros((len(LegoMetrics)))
    metrics_enum = LegoMetrics

    def __init__(self, map_shape, ctrl_metrics, n_blocks, reward):
        self.tile_enum = LegoTiles
        # self.map_shape = map_shape
        self.metrics_enum = LegoMetrics
        self.n_blocks = n_blocks

        stat_trgs = np.zeros(len(LegoMetrics))
        stat_trgs[LegoMetrics.AVG_HEIGHT] = sum([i for i in range(self.n_blocks)])/self.n_blocks
        stat_trgs[LegoMetrics.FOOTPRINT] = 1
        stat_trgs[LegoMetrics.AVG_EUCLIDEAN] = 0
        stat_trgs[LegoMetrics.CENTER] = 0
        stat_trgs[LegoMetrics.HOUSE] = 1
        stat_trgs[LegoMetrics.TABLE] = 1
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

        # TO DO: fix this
        #table_ness
        roof_x = blocks[-1,0]
        roof_z = blocks[-1,2]

        not_table = 0
        for block in blocks[:-1,:]:
            curr_x, curr_y, curr_z, _ = block
            truth_cond = jnp.logical_and(
                jnp.logical_or(
                    curr_x == roof_x,
                    curr_x == roof_x + 3
                ),
                jnp.logical_and(
                    jnp.logical_or(
                        curr_z == roof_z,
                        curr_z == roof_z + 3
                    ),
                    curr_y < blocks.shape[1]/4
                ) 
                )
            in_position = jnp.where(truth_cond, 0, 1)
            not_table += in_position
        not_table += jnp.where(blocks[-1,1] == blocks.shape[1]//4, 0, 1)
        stats = stats.at[LegoMetrics.TABLE].set(not_table == 0)
        
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
    
    