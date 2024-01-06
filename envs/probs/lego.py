from enum import IntEnum
from typing import Optional

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.utils import Tiles
from .problem import Problem

class LegoTiles(IntEnum):
    EMPTY = 0
    N3005 = 1 #1x1 piece, number 3005

@struct.dataclass
class LegoProblemState:
    stats: Optional[chex.Array] = None
    ctrl_trgs: Optional[chex.Array] = None
    

class LegoMetrics(IntEnum):
    AVG_HEIGHT = 0
    FOOTPRINT = 1 #num blocks touching the floor

class LegoProblem(Problem):
    #tile_size = np.int8(16)
    stat_weights = jnp.ones((len(LegoMetrics)))
    stat_weights = stat_weights.at[1].set(0.5)
    metrics_enum = LegoMetrics

    def __init__(self, map_shape, ctrl_metrics):
        self.tile_enum = LegoTiles
        # self.map_shape = map_shape
        self.metrics_enum = LegoMetrics
        
        stat_trgs = np.zeros(len(LegoMetrics))
        stat_trgs[LegoMetrics.AVG_HEIGHT] = np.inf
        stat_trgs[LegoMetrics.FOOTPRINT] = 1
        self.stat_trgs = jnp.array(stat_trgs)
        
        super().__init__(map_shape = map_shape, ctrl_metrics = ctrl_metrics)

    def gen_init_map(self, rng):
        return jnp.zeros(self.map_shape)
        #return gen_init_map(rng, self.tile_enum, self.map_shape,
        #                       self.tile_probs)

    def get_metric_bounds(self, map_shape):
        bounds = [None] * len(LegoMetrics)
        bounds[LegoMetrics.AVG_HEIGHT] = [0, map_shape[1]]
        bounds[LegoMetrics.FOOTPRINT] = [1, map_shape[0] * map_shape[2]]
        return np.array(bounds)

    def get_curr_stats(self, blocks: chex.Array, env_map: chex.Array = None):
        """Get relevant metrics from the current state of the environment."""
        
        total_heights = jnp.sum(blocks[:,1], axis = None)
        avg_height = total_heights/blocks.shape[0]
        #max_height = jnp.max(blocks[:,1])

        footprint = jnp.count_nonzero(jnp.where(blocks[:,1] == 0, 1, 0))
        

        stats = jnp.zeros(len(LegoMetrics))
        stats = stats.at[LegoMetrics.AVG_HEIGHT].set(avg_height)
        stats = stats.at[LegoMetrics.FOOTPRINT].set(footprint)

        state = LegoProblemState(
            stats=stats, ctrl_trgs=self.get_stat_trgs)
        
        
        return state
    





    #########################################
    def queue_ctrl_trgs(self, queued_state, ctrl_trgs):
        queued_state = queued_state.replace(queued_ctrl_trgs=ctrl_trgs, has_queued_ctrl_trgs=True)
        return queued_state
        #raise NotImplementedError

    @property
    def get_stat_trgs(self):
        return self.stat_trgs
   
    def reset(self, blocks: chex.Array):

        
        state = LegoProblemState(ctrl_trgs = self.get_stat_trgs, stats = self.get_curr_stats(blocks).stats)
        reward = 0
        return reward, state

    

    def step(self, env_map: chex.Array, state: LegoProblemState, blocks:chex.Array):
        new_state = self.get_curr_stats(blocks=blocks)
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
    
    