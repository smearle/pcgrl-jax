from enum import IntEnum
import os
from typing import Optional

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.pathfinding import FloodPath, FloodPathState, FloodRegions, FloodRegionsState, calc_diameter, get_max_n_regions, get_max_path_length
from envs.probs.problem import Problem, ProblemState, get_reward
from envs.utils import Tiles


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class BinaryTiles(IntEnum):
    BORDER = 0
    EMPTY = 1
    WALL = 2


@struct.dataclass
class BinaryState(ProblemState):
    # flood_path_state: FloodPathState
    # flood_regions_state: FloodRegionsState
    diameter: int
    n_regions: int
    # ctrl_trgs: Optional[chex.Array] = None
    flood_count: Optional[chex.Array] = None


class BinaryMetrics(IntEnum):
    DIAMETER = 0
    N_REGIONS = 1

    
def get_metric_bounds(map_shape):
    bounds = [None] * len(BinaryMetrics)
    bounds[BinaryMetrics.DIAMETER] = [0, get_max_path_length(map_shape)]
    bounds[BinaryMetrics.N_REGIONS] = [0, get_max_n_regions(map_shape)]
    return np.array(bounds)


class BinaryProblem(Problem):
    tile_enum = BinaryTiles

    tile_probs = np.zeros(len(tile_enum))
    tile_probs[BinaryTiles.EMPTY] = 0.5
    tile_probs[BinaryTiles.WALL] = 0.5
    tile_probs = jnp.array(tile_probs)

    stat_weights = np.zeros(len(BinaryMetrics))
    stat_weights[BinaryMetrics.DIAMETER] = 1.0
    stat_weights[BinaryMetrics.N_REGIONS] = 1.0
    stat_weights = jnp.array(stat_weights)

    stat_trgs = np.zeros(len(BinaryMetrics))
    stat_trgs[BinaryMetrics.DIAMETER] = np.inf
    stat_trgs[BinaryMetrics.N_REGIONS] = 1
    stat_trgs = jnp.array(stat_trgs)

    metrics_enum = BinaryMetrics

    passable_tiles = jnp.array([BinaryTiles.EMPTY])

    def __init__(self, map_shape, ctrl_metrics):
        self.flood_path_net = FloodPath()
        self.flood_path_net.init_params(map_shape)
        self.flood_regions_net = FloodRegions()
        self.flood_regions_net.init_params(map_shape)
        self.max_path_len = get_max_path_length(map_shape)
        self.metric_bounds = get_metric_bounds(map_shape)
        self.ctrl_metrics = np.array(ctrl_metrics, dtype=int)
        self.ctrl_metrics_mask = np.array([i in ctrl_metrics for i in range(len(self.stat_trgs))])
        # self.default_ctrl_trgs = [self.stat_trgs[e] for e in self.metrics_enum]
        super().__init__()
    
    def init_graphics(self):
        self.graphics = [0] * (len(self.tile_enum) + 1)
        self.graphics[BinaryTiles.EMPTY] = Image.open(
                f"{__location__}/tile_ims/empty.png"
            ).convert('RGBA')
        self.graphics[BinaryTiles.WALL] = Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA')
        self.graphics[BinaryTiles.BORDER] = Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA')
        self.graphics[len(self.tile_enum)] = Image.open(f"{__location__}/tile_ims/path_g.png").convert(
                'RGBA')
        self.graphics = jnp.array(self.graphics)

        super().init_graphics()

    def get_curr_stats(self, env_map: chex.Array):
        diameter, flood_path_state, n_regions, flood_regions_state = calc_diameter(
            self.flood_regions_net, self.flood_path_net, env_map, self.passable_tiles
        )
        stats = jnp.zeros(len(BinaryMetrics))
        stats = stats.at[BinaryMetrics.DIAMETER].set(diameter)
        stats = stats.at[BinaryMetrics.N_REGIONS].set(n_regions)
        return stats, flood_path_state.flood_count

    def reset(self, env_map: chex.Array, rng):
        stats, flood_count = self.get_curr_stats(env_map)
        reward = None
        # Randomly sample some control targets
        ctrl_trgs =  jnp.where(
            self.ctrl_metrics_mask,
            gen_ctrl_trgs(self.metric_bounds, rng),
            self.stat_trgs,
        )
        prob_state = BinaryState(
            diameter=stats[BinaryMetrics.DIAMETER],
            n_regions=stats[BinaryMetrics.N_REGIONS],
            flood_count=flood_count,
            stats=stats,
            ctrl_trgs=ctrl_trgs,
        )
        return reward, prob_state

    def step(self, env_map: chex.Array, prob_state: BinaryState):
        stats, flood_count = self.get_curr_stats(env_map)
        old_stats = prob_state.stats
        diameter, n_regions = stats[BinaryMetrics.DIAMETER], stats[BinaryMetrics.N_REGIONS]
        # reward = get_reward(stats, old_stats, self.stat_weights, prob_state.ctrl_trgs)
        # reward = get_reward_old(stats, old_stats, self.stat_weights, self.stat_trgs)
        reward = get_reward(stats, old_stats, self.stat_weights, prob_state.ctrl_trgs)
        # return reward, BinaryState(flood_path_state, flood_regions_state, diameter, n_regions)
        return reward, BinaryState(
            diameter=diameter,
            n_regions=n_regions,
            flood_count=flood_count,
            stats=stats,
            ctrl_trgs=prob_state.ctrl_trgs,
        )


def gen_ctrl_trgs(metric_bounds, rng):
    rng, _ = jax.random.split(rng)
    return jax.random.randint(rng, (len(metric_bounds),), metric_bounds[:, 0], metric_bounds[:, 1])