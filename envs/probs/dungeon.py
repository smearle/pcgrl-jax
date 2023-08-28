from enum import IntEnum
import os
from typing import Optional

import chex
from flax import struct
import jax.numpy as jnp
from PIL import Image
import numpy as np

from envs.pathfinding import FloodPath, FloodPathState, FloodRegions, FloodRegionsState, calc_diameter
from envs.probs.problem import Problem, ProblemState, get_reward
from envs.utils import Tiles


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class DungeonTiles(IntEnum):
    BORDER = 0
    EMPTY = 1
    WALL = 2
    PLAYER = 3
    BAT = 4
    SCORPION = 5
    SPIDER = 6
    KEY = 7
    DOOR = 8


class DungeonMetrics(IntEnum):
    N_REGIONS = 0
    N_ENEMIES = 1
    N_PLAYERS = 2
    N_KEYS = 3
    N_DOORS = 4
    PATH_LENGTH = 5
    NEAREST_ENEMY = 6


@struct.dataclass
class DungeonState(ProblemState):
    flood_path_state: FloodPathState
    flood_regions_state: FloodRegionsState
    diameter: int
    n_regions: int


class DungeonProblem(Problem):

    tile_enum = DungeonTiles

    tile_probs = [0.0] * len(tile_enum)
    tile_probs[DungeonTiles.EMPTY] = 0.58
    tile_probs[DungeonTiles.WALL] = 0.3
    tile_probs[DungeonTiles.PLAYER] = 0.02
    tile_probs[DungeonTiles.KEY] = 0.02
    tile_probs[DungeonTiles.DOOR] = 0.02
    tile_probs[DungeonTiles.BAT] = 0.02
    tile_probs[DungeonTiles.SCORPION] = 0.02
    tile_probs[DungeonTiles.SPIDER] = 0.02
    tile_probs = jnp.array(tile_probs)

    stat_weights = {
        DungeonMetrics.N_REGIONS: 5,
        DungeonMetrics.N_PLAYERS: 3,
        DungeonMetrics.N_KEYS: 3,
        DungeonMetrics.N_DOORS: 3,
        DungeonMetrics.NEAREST_ENEMY: 2,
        DungeonMetrics.PATH_LENGTH: 1,
    }

    stat_trgs = {
        DungeonMetrics.N_REGIONS: 1,
        DungeonMetrics.N_PLAYERS: 1,
        DungeonMetrics.N_KEYS: 1,
        DungeonMetrics.N_DOORS: 1,
        DungeonMetrics.N_ENEMIES: (2, 5),
        DungeonMetrics.PATH_LENGTH: 'max',
        DungeonMetrics.NEAREST_ENEMY: (2, np.inf),
    }

    passable_tiles = jnp.array([DungeonTiles.EMPTY, DungeonTiles.KEY,
                                DungeonTiles.SCORPION, DungeonTiles.SPIDER,
                                DungeonTiles.BAT])

    def __init__(self, map_shape):
        self.flood_path_net = FloodPath()
        self.flood_path_net.init_params(map_shape)
        self.flood_regions_net = FloodRegions()
        self.flood_regions_net.init_params(map_shape)

    def init_graphics(self):
        self.graphics = {
            DungeonTiles.EMPTY: Image.open(
                f"{__location__}/tile_ims/empty.png"
            ).convert('RGBA'),
            DungeonTiles.WALL: Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA'),
            DungeonTiles.BORDER: Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA'),
            DungeonTiles.KEY: Image.open(
                f"{__location__}/tile_ims/key.png"
            ).convert('RGBA'),
            DungeonTiles.DOOR: Image.open(
                f"{__location__}/tile_ims/door.png"
            ).convert('RGBA'),
            DungeonTiles.PLAYER: Image.open(
                f"{__location__}/tile_ims/player.png"
            ).convert('RGBA'),
            DungeonTiles.BAT: Image.open(
                f"{__location__}/tile_ims/bat.png"
            ).convert('RGBA'),
            DungeonTiles.SCORPION: Image.open(
                f"{__location__}/tile_ims/scorpion.png"
            ).convert('RGBA'),
            DungeonTiles.SPIDER: Image.open(
                f"{__location__}/tile_ims/spider.png"
            ).convert('RGBA'),
            "path": Image.open(f"{__location__}/tile_ims/path_g.png").convert(
                'RGBA'
            ),
        }

    def get_stats(self, env_map: chex.Array, prob_state: Optional[DungeonState] = None):
        diameter, flood_path_state, n_regions, flood_regions_state = calc_diameter(self.flood_regions_net, self.flood_path_net, env_map, self.passable_tiles)
        stats = {
            DungeonMetrics.DIAMETER: diameter,
            DungeonMetrics.N_REGIONS: n_regions,
        }
        if prob_state:
            last_diameter, last_n_regions = prob_state.diameter, prob_state.n_regions
            old_stats = {
                DungeonMetrics.DIAMETER: last_diameter,
                DungeonMetrics.N_REGIONS: last_n_regions,
            }
            reward = get_reward(stats, old_stats, self.stat_weights, self.stat_trgs)
        else:
            reward = None
        return reward, DungeonState(flood_path_state, flood_regions_state, diameter, n_regions)
