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

from envs.pathfinding import (FloodPath, FloodRegions, calc_n_regions, 
                              calc_path_length, get_max_n_regions, 
                              get_max_path_length, get_path_coords)
from envs.probs.problem import Problem, ProblemState
from envs.utils import Tiles


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class MazeTiles(IntEnum):
    BORDER = 0
    EMPTY = 1
    WALL = 2
    PLAYER = 3
    DOOR = 4


@struct.dataclass
class MazeState(ProblemState):
    flood_count: Optional[chex.Array] = None


class MazeMetrics(IntEnum):
    PATH_LENGTH = 0
    N_REGIONS = 1
    N_PLAYERS = 2
    N_DOORS = 3
    

class MazeProblem(Problem):
    tile_enum = MazeTiles

    tile_probs = np.zeros(len(tile_enum))
    tile_probs[MazeTiles.EMPTY] = 0.48
    tile_probs[MazeTiles.WALL] = 0.48
    tile_probs[MazeTiles.PLAYER] = 0.02
    tile_probs[MazeTiles.DOOR] = 0.02
    tile_probs = jnp.array(tile_probs)

    stat_weights = np.zeros(len(MazeMetrics))
    stat_weights[MazeMetrics.PATH_LENGTH] = 1.0
    stat_weights[MazeMetrics.N_REGIONS] = 1.0
    stat_weights[MazeMetrics.N_PLAYERS] = 1.0
    stat_weights[MazeMetrics.N_DOORS] = 1.0
    stat_weights = jnp.array(stat_weights)

    stat_trgs = np.zeros(len(MazeMetrics))
    stat_trgs[MazeMetrics.PATH_LENGTH] = np.inf
    stat_trgs[MazeMetrics.N_REGIONS] = 1
    stat_trgs[MazeMetrics.N_PLAYERS] = 1
    stat_trgs[MazeMetrics.N_DOORS] = 1
    stat_trgs = jnp.array(stat_trgs)

    metrics_enum = MazeMetrics

    passable_tiles = jnp.array([MazeTiles.EMPTY, MazeTiles.PLAYER, MazeTiles.DOOR])

    def __init__(self, map_shape, ctrl_metrics):
        self.flood_path_net = FloodPath()
        self.flood_path_net.init_params(map_shape)
        self.flood_regions_net = FloodRegions()
        self.flood_regions_net.init_params(map_shape)
        self.max_path_len = get_max_path_length(map_shape)
        super().__init__(map_shape, ctrl_metrics)

    def get_metric_bounds(self, map_shape):
        bounds = [None] * len(MazeMetrics)
        bounds[MazeMetrics.PATH_LENGTH] = [0, get_max_path_length(map_shape)]
        bounds[MazeMetrics.N_REGIONS] = [0, get_max_n_regions(map_shape)]
        bounds[MazeMetrics.N_PLAYERS] = [0, math.prod(map_shape)]
        bounds[MazeMetrics.N_DOORS] = [0, math.prod(map_shape)]
        return np.array(bounds)

    def get_path_coords(self, env_map: chex.Array, prob_state: MazeState):
        coord1 = jnp.argwhere(env_map == MazeTiles.DOOR, size=1, fill_value=-1)[0]
        return get_path_coords(prob_state.flood_count, max_path_len=self.max_path_len, coord1=coord1)

    def get_curr_stats(self, env_map: chex.Array):
        n_players = jnp.sum(env_map == MazeTiles.PLAYER)
        n_doors = jnp.sum(env_map == MazeTiles.DOOR)
        n_regions = calc_n_regions(self.flood_regions_net, env_map, self.passable_tiles)
        path_length, flood_count, _ = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(n_players == 1, n_doors == 1), n_regions == 1),
            lambda: calc_path_length(
                flood_path_net=self.flood_path_net, 
                env_map=env_map, 
                passable_tiles=self.passable_tiles, 
                src=MazeTiles.PLAYER, trg=MazeTiles.DOOR),
            lambda: (0.0, jnp.zeros(env_map.shape, dtype=jnp.float32), jnp.zeros(2, dtype=jnp.int32))
        )
        stats = jnp.zeros(len(MazeMetrics))
        stats = stats.at[MazeMetrics.PATH_LENGTH].set(path_length)
        stats = stats.at[MazeMetrics.N_REGIONS].set(n_regions)
        stats = stats.at[MazeMetrics.N_PLAYERS].set(n_players)
        stats = stats.at[MazeMetrics.N_DOORS].set(n_doors)
        state = MazeState(
            stats=stats,
            flood_count=flood_count,
            ctrl_trgs=None,
        )
        return state

    def init_graphics(self):
        self.graphics = [0] * (len(self.tile_enum) + 1)
        self.graphics[MazeTiles.EMPTY] = Image.open(
                f"{__location__}/tile_ims/empty.png"
            ).convert('RGBA')
        self.graphics[MazeTiles.WALL] = Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA')
        self.graphics[MazeTiles.BORDER] = Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA')
        self.graphics[MazeTiles.PLAYER] = Image.open(
                f"{__location__}/tile_ims/player.png"
            ).convert('RGBA')
        self.graphics[MazeTiles.DOOR] = Image.open(
                f"{__location__}/tile_ims/door.png"
            ).convert('RGBA')
        self.graphics[len(self.tile_enum)] = Image.open(f"{__location__}/tile_ims/path_g.png").convert(
                'RGBA')
        super().init_graphics()