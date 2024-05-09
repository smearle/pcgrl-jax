from enum import IntEnum
import math
import os
from typing import Optional

import chex
from flax import struct
import jax
import jax.numpy as jnp
from PIL import Image
import numpy as np

from envs.pathfinding import FloodPath, FloodPathState, FloodRegions, FloodRegionsState, calc_diameter, calc_n_regions, calc_path_length, get_max_n_regions, get_max_path_length, get_max_path_length_static, get_path_coords
from envs.probs.problem import Problem, ProblemState, draw_path, get_max_loss, get_reward
from envs.utils import idx_dict_to_arr, Tiles


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
    player_key_flood_count: Optional[chex.Array] = None
    key_door_flood_count: Optional[chex.Array] = None
    player_enemy_flood_count: Optional[chex.Array] = None
    enemy_xy: Optional[chex.Array] = None
    k_xy: Optional[chex.Array] = None
    d_xy: Optional[chex.Array] = None


class DungeonProblem(Problem):
    tile_enum = DungeonTiles
    metrics_enum = DungeonMetrics
    ctrl_threshes = np.zeros(len(DungeonMetrics))

    # tile_probs = [0.0] * len(tile_enum)
    tile_probs = {
        DungeonTiles.BORDER: 0.0,
        DungeonTiles.EMPTY: 0.58,
        DungeonTiles.WALL: 0.3,
        DungeonTiles.PLAYER: 0.02,
        DungeonTiles.KEY: 0.02,
        DungeonTiles.DOOR: 0.02,
        DungeonTiles.BAT: 0.02,
        DungeonTiles.SCORPION: 0.02,
        DungeonTiles.SPIDER: 0.02,
    }
    tile_probs = tuple(idx_dict_to_arr(tile_probs))

    tile_nums = [0 for _ in range(len(tile_enum))]
    tile_nums[DungeonTiles.PLAYER] = 1
    tile_nums[DungeonTiles.DOOR] = 1
    tile_nums[DungeonTiles.KEY] = 1
    tile_nums = tuple(tile_nums)

    stat_weights = {
        DungeonMetrics.N_REGIONS: 5,
        DungeonMetrics.N_PLAYERS: 3,
        DungeonMetrics.N_ENEMIES: 3,
        DungeonMetrics.N_KEYS: 3,
        DungeonMetrics.N_DOORS: 3,
        DungeonMetrics.NEAREST_ENEMY: 2,
        DungeonMetrics.PATH_LENGTH: 1,
    }
    stat_weights = idx_dict_to_arr(stat_weights)

    stat_trgs = {
        DungeonMetrics.N_REGIONS: 1,
        DungeonMetrics.N_PLAYERS: 1,
        DungeonMetrics.N_KEYS: 1,
        DungeonMetrics.N_DOORS: 1,
        # DungeonMetrics.N_ENEMIES: (2, 5),
        DungeonMetrics.N_ENEMIES: 3.5,
        DungeonMetrics.PATH_LENGTH: 'max',
        DungeonMetrics.NEAREST_ENEMY: (2, np.inf),
    }

    passable_tiles = jnp.array([DungeonTiles.EMPTY, DungeonTiles.KEY])
                                # DungeonTiles.SCORPION, DungeonTiles.SPIDER,
                                # DungeonTiles.BAT])

    def __init__(self, map_shape, ctrl_metrics, pinpoints, num_agents=1):
        self.flood_path_net = FloodPath()
        self.flood_path_net.init_params(map_shape)
        self.flood_regions_net = FloodRegions()
        self.flood_regions_net.init_params(map_shape)
        self.max_path_len = get_max_path_length_static(map_shape)
        self.n_tiles = math.prod(map_shape)
        self.n_agents = num_agents

        # Note that we implement target intervals by using target floats
        # and thresholds (i.e. allowable margins of error around the target
        # resulting in equivalent reward). To implement an unbounded range (e.g.
        # [0, inf]), we use some large upper bound on the metric value as target.
        stat_trgs = {
            DungeonMetrics.N_REGIONS: 1,
            DungeonMetrics.N_PLAYERS: 1,
            DungeonMetrics.N_KEYS: 1,
            DungeonMetrics.N_DOORS: 1,
            # DungeonMetrics.N_ENEMIES: (2, 5),
            DungeonMetrics.N_ENEMIES: 3.5,
            DungeonMetrics.PATH_LENGTH: np.inf,
            # DungeonMetrics.NEAREST_ENEMY: (2, np.inf),

            # FIXME: This should be max_path_len
            DungeonMetrics.NEAREST_ENEMY: self.n_tiles,
        }
        self.stat_trgs = idx_dict_to_arr(stat_trgs)
        self.ctrl_threshes[DungeonMetrics.N_ENEMIES] = 3
        self.ctrl_threshes[DungeonMetrics.NEAREST_ENEMY] = self.n_tiles - 2

        if self.n_agents > 1:
            
            stat_weights = {
                DungeonMetrics.N_REGIONS: jnp.array([5]*self.n_agents),
                DungeonMetrics.N_PLAYERS: jnp.array([3]*self.n_agents),
                DungeonMetrics.N_ENEMIES: jnp.array([3]*self.n_agents),
                DungeonMetrics.N_KEYS: jnp.array([3]*self.n_agents),
                DungeonMetrics.N_DOORS: jnp.array([3]*self.n_agents),
                DungeonMetrics.NEAREST_ENEMY: jnp.array([2]*self.n_agents),
                DungeonMetrics.PATH_LENGTH: jnp.array([1]*self.n_agents),
            }
            self.stat_weights = idx_dict_to_arr(stat_weights)


            stat_trgs = {
                DungeonMetrics.N_REGIONS: jnp.array([1]*self.n_agents),
                DungeonMetrics.N_PLAYERS: jnp.array([1]*self.n_agents),
                DungeonMetrics.N_KEYS: jnp.array([1]*self.n_agents),
                DungeonMetrics.N_DOORS: jnp.array([1]*self.n_agents),
                # DungeonMetrics.N_ENEMIES: (2, 5),
                DungeonMetrics.N_ENEMIES: jnp.array([3.5]*self.n_agents),
                DungeonMetrics.PATH_LENGTH: jnp.array([np.inf]*self.n_agents),
                # DungeonMetrics.NEAREST_ENEMY: (2, np.inf),

                # FIXME: This should be max_path_len
                DungeonMetrics.NEAREST_ENEMY: jnp.array([self.n_tiles]*self.n_agents),
            }

            self.stat_trgs = idx_dict_to_arr(stat_trgs) 

        super().__init__(map_shape=map_shape, ctrl_metrics=ctrl_metrics, pinpoints=pinpoints)

    def get_metric_bounds(self, map_shape):
        bounds = [None] * len(DungeonMetrics)
        bounds[DungeonMetrics.PATH_LENGTH] = [0, self.max_path_len * 2]
        bounds[DungeonMetrics.N_PLAYERS] = [0, self.n_tiles]
        bounds[DungeonMetrics.N_KEYS] = [0, self.n_tiles]
        bounds[DungeonMetrics.N_DOORS] = [0, self.n_tiles]
        bounds[DungeonMetrics.N_ENEMIES] = [0, self.n_tiles]
        bounds[DungeonMetrics.N_REGIONS] = [0, self.n_tiles]
        bounds[DungeonMetrics.NEAREST_ENEMY] = [0, self.max_path_len]
        return jnp.array(bounds)

    def get_path_coords(self, env_map: chex.Array, prob_state: ProblemState):
        """Return a list of tile coords, starting somewhere (assumed in the flood) and following the water level upward."""
        # k_xy = jnp.argwhere(env_map == DungeonTiles.KEY, size=1, fill_value=-1)[0]
        # d_xy = jnp.argwhere(env_map == DungeonTiles.DOOR, size=1, fill_value=-1)[0]
        k_xy = prob_state.k_xy
        d_xy = prob_state.d_xy
        e_xy = prob_state.enemy_xy
        pk_flood_count = prob_state.player_key_flood_count
        kd_flood_count = prob_state.key_door_flood_count
        pe_flood_count = prob_state.player_enemy_flood_count
        pk_coords = get_path_coords(pk_flood_count, max_path_len=self.max_path_len, coord1=k_xy)
        kd_coords = get_path_coords(kd_flood_count, max_path_len=self.max_path_len, coord1=d_xy)
        pe_coords = get_path_coords(pe_flood_count, max_path_len=self.max_path_len, coord1=e_xy) 
        return (pk_coords, kd_coords, pe_coords)

    def get_curr_stats(self, env_map: chex.Array):
        n_players = jnp.sum(env_map == DungeonTiles.PLAYER)
        n_doors = jnp.sum(env_map == DungeonTiles.DOOR)
        n_keys = jnp.sum(env_map == DungeonTiles.KEY)
        n_enemies = jnp.sum(
            (env_map == DungeonTiles.BAT) | (env_map == DungeonTiles.SCORPION) | (env_map == DungeonTiles.SPIDER))
        n_regions, _ = calc_n_regions(self.flood_regions_net, env_map, self.passable_tiles)
        is_playable: bool = (n_players == 1) & (n_doors == 1) & (n_keys == 1) & (5 >= n_enemies) & (n_enemies >= 2) & (n_regions == 1)

        # Get path from player to key and from key to door
        pk_path_length, pk_flood_count, k_xy = jax.lax.cond(
            is_playable,
            lambda: calc_path_length(
                flood_path_net=self.flood_path_net, 
                env_map=env_map, 
                passable_tiles=self.passable_tiles, 
                src=DungeonTiles.PLAYER, trg=DungeonTiles.KEY),
            lambda: (0.0, jnp.zeros(env_map.shape, dtype=jnp.float32), jnp.full(2, dtype=jnp.int32, fill_value=-1))
        )
        kd_passable_tiles = jnp.concatenate((self.passable_tiles, jnp.array([DungeonTiles.DOOR])))
        kd_path_length, kd_flood_count, d_xy = jax.lax.cond(
            is_playable,
            lambda: calc_path_length(
                flood_path_net=self.flood_path_net, 
                env_map=env_map, 
                passable_tiles=kd_passable_tiles, 
                src=DungeonTiles.KEY, trg=DungeonTiles.DOOR),
            lambda: (0.0, jnp.zeros(env_map.shape, dtype=jnp.float32), jnp.full(2, dtype=jnp.int32, fill_value=-1))
        )
        path_length = pk_path_length + kd_path_length

        # Encode all enemies as bat tiles to make pathfinding simpler
        env_map_uni_enemy = jnp.where(
                ((env_map == DungeonTiles.BAT) | (env_map == DungeonTiles.SCORPION) | (env_map == DungeonTiles.SPIDER)) > 0,
            DungeonTiles.BAT, env_map)

        # Get path length from player to nearest enemy
        pe_passable_tiles = jnp.concatenate((self.passable_tiles, jnp.array([DungeonTiles.BAT])))
        pe_path_length, pe_flood_count, e_xy = jax.lax.cond(
            is_playable,
            lambda: calc_path_length(
                flood_path_net=self.flood_path_net, 
                env_map=env_map_uni_enemy, 
                passable_tiles=pe_passable_tiles, 
                src=DungeonTiles.PLAYER, trg=DungeonTiles.BAT),
            lambda: (0.0, jnp.zeros(env_map.shape, dtype=jnp.float32), jnp.full(2, dtype=jnp.int32, fill_value=-1))
        )

        stats = jnp.zeros(len(DungeonMetrics))
        stats = stats.at[DungeonMetrics.PATH_LENGTH].set(path_length)
        stats = stats.at[DungeonMetrics.N_ENEMIES].set(n_enemies)
        stats = stats.at[DungeonMetrics.N_KEYS].set(n_keys)
        stats = stats.at[DungeonMetrics.NEAREST_ENEMY].set(pe_path_length)
        stats = stats.at[DungeonMetrics.N_REGIONS].set(n_regions)
        stats = stats.at[DungeonMetrics.N_PLAYERS].set(n_players)
        stats = stats.at[DungeonMetrics.N_DOORS].set(n_doors)
        state = DungeonState(
            stats=stats,
            player_key_flood_count=pk_flood_count,
            key_door_flood_count=kd_flood_count,
            player_enemy_flood_count=pe_flood_count,
            enemy_xy=e_xy,
            k_xy=k_xy,
            d_xy=d_xy,
            ctrl_trgs=None,
        )
        return state

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
            len(DungeonTiles): Image.open(f"{__location__}/tile_ims/path_g.png").convert(
                'RGBA'
            ),
            len(DungeonTiles) + 1: Image.open(f"{__location__}/tile_ims/path_purple.png").convert(
                'RGBA'
            )
        }
        self.graphics = jnp.array(idx_dict_to_arr(self.graphics))
        super().init_graphics()

    def draw_path(self, lvl_img,env_map, border_size, path_coords_tpl, tile_size):
        assert len(path_coords_tpl) == 3
        lvl_img = draw_path(prob=self, lvl_img=lvl_img, env_map=env_map, border_size=border_size,
                            path_coords=path_coords_tpl[0], tile_size=tile_size,
                            im_idx=-2)
        lvl_img = draw_path(prob=self, lvl_img=lvl_img, env_map=env_map, border_size=border_size,
                            path_coords=path_coords_tpl[1], tile_size=tile_size,
                            im_idx=-2)
        lvl_img = draw_path(prob=self, lvl_img=lvl_img, env_map=env_map, border_size=border_size,
                            path_coords=path_coords_tpl[2], tile_size=tile_size)
        return lvl_img

