from enum import IntEnum
import os
from typing import Optional, Tuple

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.pathfinding import FloodPath, FloodPathState, FloodRegions, FloodRegionsState, calc_diameter, get_max_n_regions, get_max_path_length, get_max_path_length_static, get_path_coords_diam
from envs.probs.problem import Problem, ProblemState, get_reward
from envs.utils import Tiles


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class BinaryTiles(IntEnum):
    BORDER = 0
    EMPTY = 1
    WALL = 2


@struct.dataclass
class BinaryState(ProblemState):
    flood_count: Optional[chex.Array] = None


class BinaryMetrics(IntEnum):
    DIAMETER = 0
    N_REGIONS = 1

    

class BinaryProblem(Problem):
    tile_enum = BinaryTiles

    tile_probs = np.zeros(len(tile_enum))
    tile_probs[BinaryTiles.EMPTY] = 0.5
    tile_probs[BinaryTiles.WALL] = 0.5
    tile_probs = tuple(tile_probs)

    tile_nums = tuple([0 for _ in range(len(tile_enum))])

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

    def __init__(self, map_shape, ctrl_metrics, pinpoints):
        self.flood_path_net = FloodPath()
        self.flood_path_net.init_params(map_shape)
        self.flood_regions_net = FloodRegions()
        self.flood_regions_net.init_params(map_shape)
        self.max_path_len = get_max_path_length_static(map_shape)
        super().__init__(map_shape=map_shape, ctrl_metrics=ctrl_metrics, pinpoints=pinpoints)

    def get_metric_bounds(self, map_shape):
        bounds = [None] * len(BinaryMetrics)
        bounds[BinaryMetrics.DIAMETER] = [0, get_max_path_length(map_shape)]
        bounds[BinaryMetrics.N_REGIONS] = [0, get_max_n_regions(map_shape)]
        return jnp.array(bounds)

    def get_path_coords(self, env_map: chex.Array, prob_state: BinaryState) -> Tuple[chex.Array]:
        return (get_path_coords_diam(flood_count=prob_state.flood_count, max_path_len=self.max_path_len),)
    
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

        super().init_graphics()

    def get_curr_stats(self, env_map: chex.Array):
        """Get relevant metrics from the current state of the environment."""
        diameter, flood_path_state, n_regions, flood_regions_state = calc_diameter(
            self.flood_regions_net, self.flood_path_net, env_map, self.passable_tiles
        )
        stats = jnp.zeros(len(BinaryMetrics))
        stats = stats.at[BinaryMetrics.DIAMETER].set(diameter)
        stats = stats.at[BinaryMetrics.N_REGIONS].set(n_regions)
        state = BinaryState(
            stats=stats, flood_count=flood_path_state.flood_count, ctrl_trgs=None)
        return state
