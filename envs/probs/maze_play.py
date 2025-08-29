from enum import IntEnum
import math
import os
from typing import Optional

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.pathfinding import FloodPath, FloodPathState, FloodRegions, calc_n_regions, calc_path_length, get_max_n_regions, get_max_path_length, get_path_coords
from envs.probs.problem import Problem, ProblemState, get_reward
from envs.utils import Tiles


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class MazePlayTiles(IntEnum):
    BORDER = 0
    EMPTY = 1
    WALL = 2
    PLAYER = 3
    DOOR = 4


@struct.dataclass
class MazePlayState(ProblemState):
    # player_xy: chex.Array = None
    pass


class MazePlayMetrics(IntEnum):
    IS_SOLVED = 0
    

class MazePlayProblem(Problem):
    tile_enum = MazePlayTiles

    tile_probs = np.zeros(len(tile_enum))
    tile_probs[MazePlayTiles.EMPTY] = 0.48
    tile_probs[MazePlayTiles.WALL] = 0.48
    tile_probs[MazePlayTiles.PLAYER] = 0.02
    tile_probs[MazePlayTiles.DOOR] = 0.02
    tile_probs = jnp.array(tile_probs)

    stat_weights = np.zeros(len(MazePlayMetrics))
    stat_weights[MazePlayMetrics.IS_SOLVED] = 1.0
    stat_weights = jnp.array(stat_weights)

    stat_trgs = np.zeros(len(MazePlayMetrics))
    stat_trgs[MazePlayMetrics.IS_SOLVED] = 1
    stat_trgs = jnp.array(stat_trgs)

    metrics_enum = MazePlayMetrics

    passable_tiles = (MazePlayTiles.EMPTY, MazePlayTiles.PLAYER, MazePlayTiles.DOOR)

    def __init__(self, map_shape, ctrl_metrics):
        self.flood_path_net = FloodPath()
        self.flood_path_net.init_params(map_shape)
        self.flood_regions_net = FloodRegions()
        self.flood_regions_net.init_params(map_shape)
        self.max_path_len = get_max_path_length(map_shape)
        self.map_shape = map_shape
        super().__init__(map_shape, ctrl_metrics)

    def get_metric_bounds(self, map_shape):
        bounds = [None] * len(MazePlayMetrics)
        bounds[MazePlayMetrics.IS_SOLVED] = [0, 1]
        return np.array(bounds)

    def get_path_coords(self, env_map: chex.Array, prob_state: MazePlayState):
        return jnp.full((1,2), -1)

    def get_curr_stats(self, env_map: chex.Array, pos):
        is_solved = env_map[tuple(pos)] == MazePlayTiles.DOOR
        stats = jnp.zeros(len(MazePlayMetrics))
        stats = stats.at[MazePlayMetrics.IS_SOLVED].set(is_solved)
        state = MazePlayState(
            stats=stats,
            ctrl_trgs=None,
        )
        return state

    def init_graphics(self):
        self.graphics = [0] * (len(self.tile_enum) + 1)
        self.graphics[MazePlayTiles.EMPTY] = Image.open(
                f"{__location__}/tile_ims/empty.png"
            ).convert('RGBA')
        self.graphics[MazePlayTiles.WALL] = Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA')
        self.graphics[MazePlayTiles.BORDER] = Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA')
        self.graphics[MazePlayTiles.PLAYER] = Image.open(
                f"{__location__}/tile_ims/player.png"
            ).convert('RGBA')
        self.graphics[MazePlayTiles.DOOR] = Image.open(
                f"{__location__}/tile_ims/door.png"
            ).convert('RGBA')
        self.graphics[len(self.tile_enum)] = Image.open(f"{__location__}/tile_ims/path_g.png").convert(
                'RGBA')
        super().init_graphics()