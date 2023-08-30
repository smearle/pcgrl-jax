from enum import IntEnum
import os
from typing import Optional

import chex
from flax import struct
import jax.numpy as jnp
from PIL import Image

from envs.pathfinding import FloodPath, FloodPathState, FloodRegions, FloodRegionsState, calc_diameter, get_max_path_length
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
    flood_count: Optional[chex.Array] = None


class BinaryMetrics(IntEnum):
    DIAMETER = 0
    N_REGIONS = 1


class BinaryProblem(Problem):

    tile_enum = BinaryTiles

    tile_probs = [0.0] * len(tile_enum)
    tile_probs[BinaryTiles.EMPTY] = 0.5
    tile_probs[BinaryTiles.WALL] = 0.5
    tile_probs = jnp.array(tile_probs)

    stat_weights = {
        BinaryMetrics.DIAMETER: 1,
        BinaryMetrics.N_REGIONS: 0,
    }

    stat_trgs = {
        BinaryMetrics.DIAMETER: 'max',
        BinaryMetrics.N_REGIONS: 1,
    }

    passable_tiles = jnp.array([BinaryTiles.EMPTY])

    def __init__(self, map_shape):
        self.flood_path_net = FloodPath()
        self.flood_path_net.init_params(map_shape)
        self.flood_regions_net = FloodRegions()
        self.flood_regions_net.init_params(map_shape)
        self.max_path_len = get_max_path_length(map_shape)
    
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

    def get_stats(self, env_map: chex.Array, prob_state: Optional[BinaryState] = None):
        diameter, flood_path_state, n_regions, flood_regions_state = calc_diameter(
            self.flood_regions_net, self.flood_path_net, env_map, self.passable_tiles
        )
        stats = {
            BinaryMetrics.DIAMETER: diameter,
            BinaryMetrics.N_REGIONS: n_regions,
        }
        if prob_state:
            last_diameter, last_n_regions = prob_state.diameter, prob_state.n_regions
            old_stats = {
                BinaryMetrics.DIAMETER: last_diameter,
                BinaryMetrics.N_REGIONS: last_n_regions,
            }
            reward = get_reward(stats, old_stats, self.stat_weights, self.stat_trgs)
        else:
            reward = None
        # return reward, BinaryState(flood_path_state, flood_regions_state, diameter, n_regions)
        return reward, BinaryState(diameter, n_regions, flood_count=flood_path_state.flood_count)
