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


class Dungeon2Tiles(IntEnum):
    BORDER = 0
    EMPTY = 1
    WALL = 2
    PLAYER = 3
    BAT = 4
    SCORPION = 5
    SPIDER = 6
    KEY = 7
    DOOR = 8
    TREASURE = 9


class Dungeon2Metrics(IntEnum):
    N_REGIONS = 0
    N_ENEMIES = 1
    N_PLAYERS = 2
    N_KEYS = 3
    N_DOORS = 4
    PATH_LENGTH = 5
    NEAREST_ENEMY = 6
    N_TREASURES = 7
    DOOR_ON_THRESHOLD = 8
    TREASURE_IN_ROOM = 9


adj_mask = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
])


@struct.dataclass
class Dungeon2State(ProblemState):
    player_key_flood_count: Optional[chex.Array] = None
    key_door_flood_count: Optional[chex.Array] = None
    door_treasure_flood_count: Optional[chex.Array] = None
    player_enemy_flood_count: Optional[chex.Array] = None
    enemy_xy: Optional[chex.Array] = None
    k_xy: Optional[chex.Array] = None
    d_xy: Optional[chex.Array] = None
    t_xy: Optional[chex.Array] = None


class Dungeon2Problem(Problem):
    tile_enum = Dungeon2Tiles
    metrics_enum = Dungeon2Metrics
    ctrl_threshes = np.zeros(len(Dungeon2Metrics))

    # tile_probs = [0.0] * len(tile_enum)
    tile_probs = {
        Dungeon2Tiles.BORDER: 0.0,
        Dungeon2Tiles.EMPTY: 0.58,
        Dungeon2Tiles.WALL: 0.3,
        Dungeon2Tiles.PLAYER: 0.02,
        Dungeon2Tiles.KEY: 0.02,
        Dungeon2Tiles.DOOR: 0.01,
        Dungeon2Tiles.TREASURE: 0.01,
        Dungeon2Tiles.BAT: 0.02,
        Dungeon2Tiles.SCORPION: 0.02,
        Dungeon2Tiles.SPIDER: 0.02,
    }
    tile_probs = tuple(idx_dict_to_arr(tile_probs))

    tile_nums = [0 for _ in range(len(tile_enum))]
    tile_nums[Dungeon2Tiles.PLAYER] = 1
    tile_nums[Dungeon2Tiles.DOOR] = 1
    tile_nums[Dungeon2Tiles.KEY] = 1
    tile_nums[Dungeon2Tiles.TREASURE] = 1
    tile_nums = tuple(tile_nums)

    stat_weights = {
        Dungeon2Metrics.N_REGIONS: 5,
        Dungeon2Metrics.N_PLAYERS: 3,
        Dungeon2Metrics.N_ENEMIES: 3,
        Dungeon2Metrics.N_KEYS: 3,
        Dungeon2Metrics.N_DOORS: 3,
        Dungeon2Metrics.N_TREASURES: 3,
        Dungeon2Metrics.NEAREST_ENEMY: 2,
        Dungeon2Metrics.PATH_LENGTH: 1,
        Dungeon2Metrics.DOOR_ON_THRESHOLD: 5,
        Dungeon2Metrics.TREASURE_IN_ROOM: 5,
    }
    stat_weights = idx_dict_to_arr(stat_weights)

    stat_trgs = {
        Dungeon2Metrics.N_REGIONS: 1,
        Dungeon2Metrics.N_PLAYERS: 1,
        Dungeon2Metrics.N_KEYS: 1,
        Dungeon2Metrics.N_DOORS: 1,
        Dungeon2Metrics.N_TREASURES: 1,
        # DungeonMetrics.N_ENEMIES: (2, 5),
        Dungeon2Metrics.N_ENEMIES: 3.5,
        Dungeon2Metrics.PATH_LENGTH: 'max',
        Dungeon2Metrics.NEAREST_ENEMY: (2, np.inf),
        Dungeon2Metrics.DOOR_ON_THRESHOLD: 1,
        Dungeon2Metrics.TREASURE_IN_ROOM: 1,
    }

    passable_tiles = (Dungeon2Tiles.EMPTY, Dungeon2Tiles.KEY)
                                # DungeonTiles.SCORPION, DungeonTiles.SPIDER,
                                # DungeonTiles.BAT])

    def __init__(self, map_shape, ctrl_metrics, pinpoints):
        self.flood_path_net = FloodPath()
        self.flood_path_net.init_params(map_shape)
        self.flood_regions_net = FloodRegions()
        self.flood_regions_net.init_params(map_shape)
        self.max_path_len = get_max_path_length_static(map_shape)
        self.n_tiles = math.prod(map_shape)

        # Note that we implement target intervals by using target floats
        # and thresholds (i.e. allowable margins of error around the target
        # resulting in equivalent reward). To implement an unbounded range (e.g.
        # [0, inf]), we use some large upper bound on the metric value as target.
        stat_trgs = {
            Dungeon2Metrics.N_REGIONS: 2,
            Dungeon2Metrics.N_PLAYERS: 1,
            Dungeon2Metrics.N_KEYS: 1,
            Dungeon2Metrics.N_DOORS: 1,
            Dungeon2Metrics.N_TREASURES: 1,
            # DungeonMetrics.N_ENEMIES: (2, 5),
            Dungeon2Metrics.N_ENEMIES: 3.5,
            Dungeon2Metrics.PATH_LENGTH: np.inf,
            # DungeonMetrics.NEAREST_ENEMY: (2, np.inf),

            # FIXME: This should be max_path_len
            Dungeon2Metrics.NEAREST_ENEMY: self.max_path_len,

            Dungeon2Metrics.DOOR_ON_THRESHOLD: 1,
            Dungeon2Metrics.TREASURE_IN_ROOM: 1,
        }
        self.stat_trgs = idx_dict_to_arr(stat_trgs)
        self.ctrl_threshes[Dungeon2Metrics.N_ENEMIES] = 3
        self.ctrl_threshes[Dungeon2Metrics.NEAREST_ENEMY] = self.max_path_len - 2

        super().__init__(map_shape=map_shape, ctrl_metrics=ctrl_metrics, pinpoints=pinpoints)

    def get_metric_bounds(self, map_shape):
        bounds = [None] * len(Dungeon2Metrics)
        bounds[Dungeon2Metrics.PATH_LENGTH] = [0, self.max_path_len * 2]
        bounds[Dungeon2Metrics.N_PLAYERS] = [0, self.n_tiles]
        bounds[Dungeon2Metrics.N_KEYS] = [0, self.n_tiles]
        bounds[Dungeon2Metrics.N_DOORS] = [0, self.n_tiles]
        bounds[Dungeon2Metrics.N_TREASURES] = [0, self.n_tiles]
        bounds[Dungeon2Metrics.N_ENEMIES] = [0, self.n_tiles]
        bounds[Dungeon2Metrics.N_REGIONS] = [0, self.n_tiles]
        bounds[Dungeon2Metrics.NEAREST_ENEMY] = [0, self.max_path_len]
        bounds[Dungeon2Metrics.DOOR_ON_THRESHOLD] = [0, 1]
        bounds[Dungeon2Metrics.TREASURE_IN_ROOM] = [0, 1]
        return jnp.array(bounds)

    def get_path_coords(self, env_map: chex.Array, prob_state: Dungeon2State):
        """Return a list of tile coords, starting somewhere (assumed in the flood) and following the water level upward."""
        # k_xy = jnp.argwhere(env_map == DungeonTiles.KEY, size=1, fill_value=-1)[0]
        # d_xy = jnp.argwhere(env_map == DungeonTiles.DOOR, size=1, fill_value=-1)[0]
        k_xy = prob_state.k_xy
        d_xy = prob_state.d_xy
        t_xy = prob_state.t_xy
        e_xy = prob_state.enemy_xy
        pk_flood_count = prob_state.player_key_flood_count
        kd_flood_count = prob_state.key_door_flood_count
        dt_flood_count = prob_state.door_treasure_flood_count
        pe_flood_count = prob_state.player_enemy_flood_count
        pk_coords = get_path_coords(pk_flood_count, max_path_len=self.max_path_len, coord1=k_xy)
        kd_coords = get_path_coords(kd_flood_count, max_path_len=self.max_path_len, coord1=d_xy)
        dt_coords = get_path_coords(dt_flood_count, max_path_len=self.max_path_len, coord1=t_xy)
        pe_coords = get_path_coords(pe_flood_count, max_path_len=self.max_path_len, coord1=e_xy) 
        return (pk_coords, kd_coords, dt_coords, pe_coords)

    def get_curr_stats(self, env_map: chex.Array):
        n_players = jnp.sum(env_map == Dungeon2Tiles.PLAYER)
        n_doors = jnp.sum(env_map == Dungeon2Tiles.DOOR)
        n_keys = jnp.sum(env_map == Dungeon2Tiles.KEY)
        n_treasures = jnp.sum(env_map == Dungeon2Tiles.TREASURE)
        n_enemies = jnp.sum(
            (env_map == Dungeon2Tiles.BAT) | (env_map == Dungeon2Tiles.SCORPION) | (env_map == Dungeon2Tiles.SPIDER))
        n_regions, regions_flood = calc_n_regions(self.flood_regions_net, env_map, self.passable_tiles)

        def is_door_on_threshold():
            padded_regions_flood = jnp.pad(regions_flood, ((1, 1), (1, 1)), constant_values=0)
            d_xy = jnp.argwhere(env_map == Dungeon2Tiles.DOOR, size=1, fill_value=-1)[0]
            # adj_to_door = padded_regions_flood[d_xy[0]-1: d_xy[0]+2, d_xy[1]-1: d_xy[1]+2]
            door_patch = jax.lax.dynamic_slice(padded_regions_flood, (d_xy[0]-1, d_xy[1]-1), (3, 3))
            adj_to_door = door_patch * adj_mask
            return jnp.clip(jnp.unique(adj_to_door, size=4, fill_value=0), 0, 1).sum() >= 2

        door_is_on_threshold = jax.lax.cond(
            n_doors == 1,
            lambda: is_door_on_threshold(),
            lambda: False,
        )

        def is_treasure_in_room():
            p_xy = jnp.argwhere(env_map == Dungeon2Tiles.PLAYER, size=1, fill_value=-1)[0]
            t_xy = jnp.argwhere(env_map == Dungeon2Tiles.TREASURE, size=1, fill_value=-1)[0]
            p_region_idx = regions_flood[p_xy[0], p_xy[1]]
            t_region_idx = regions_flood[t_xy[0], t_xy[1]]
            return p_region_idx != t_region_idx

        treasure_in_room = jax.lax.cond(
            jnp.logical_and(n_treasures == 1, jnp.logical_and(n_regions > 1, n_players == 1)),
            lambda: is_treasure_in_room(),
            lambda: False,
        )

        is_playable: bool = ((n_players == 1) & (n_doors == 1) & (n_keys == 1) & (5 >= n_enemies) & (n_enemies >= 2) &
                             (n_regions <= 2) & (n_treasures == 1) & door_is_on_threshold & treasure_in_room)

        # Get path from player to key and from key to door
        pk_path_length, pk_flood_count, k_xy = jax.lax.cond(
            is_playable,
            lambda: calc_path_length(
                flood_path_net=self.flood_path_net, 
                env_map=env_map, 
                passable_tiles=self.passable_tiles, 
                src=Dungeon2Tiles.PLAYER, trg=Dungeon2Tiles.KEY),
            lambda: (0.0, jnp.zeros(env_map.shape, dtype=jnp.float32), jnp.full(2, dtype=jnp.int32, fill_value=-1))
        )
        kd_passable_tiles = jnp.concatenate((self.passable_tiles, jnp.array([Dungeon2Tiles.DOOR])))
        kd_path_length, kd_flood_count, d_xy = jax.lax.cond(
            is_playable,
            lambda: calc_path_length(
                flood_path_net=self.flood_path_net, 
                env_map=env_map, 
                passable_tiles=kd_passable_tiles, 
                src=Dungeon2Tiles.KEY, trg=Dungeon2Tiles.DOOR),
            lambda: (0.0, jnp.zeros(env_map.shape, dtype=jnp.float32), jnp.full(2, dtype=jnp.int32, fill_value=-1))
        )
        dt_passable_tiles = jnp.concatenate((kd_passable_tiles, jnp.array([Dungeon2Tiles.TREASURE])))
        dt_path_length, dt_flood_count, t_xy = jax.lax.cond(
            is_playable,
            lambda: calc_path_length(
                flood_path_net=self.flood_path_net, 
                env_map=env_map, 
                passable_tiles=dt_passable_tiles, 
                src=Dungeon2Tiles.DOOR, trg=Dungeon2Tiles.TREASURE),
            lambda: (0.0, jnp.zeros(env_map.shape, dtype=jnp.float32), jnp.full(2, dtype=jnp.int32, fill_value=-1))
        )
        path_length = pk_path_length + kd_path_length + dt_path_length

        # Encode all enemies as bat tiles to make pathfinding simpler
        env_map_uni_enemy = jnp.where(
                ((env_map == Dungeon2Tiles.BAT) | (env_map == Dungeon2Tiles.SCORPION) | (env_map == Dungeon2Tiles.SPIDER)) > 0,
            Dungeon2Tiles.BAT, env_map)

        # Get path length from player to nearest enemy
        pe_passable_tiles = jnp.concatenate((self.passable_tiles, jnp.array([Dungeon2Tiles.BAT])))
        pe_path_length, pe_flood_count, e_xy = jax.lax.cond(
            is_playable,
            lambda: calc_path_length(
                flood_path_net=self.flood_path_net, 
                env_map=env_map_uni_enemy, 
                passable_tiles=pe_passable_tiles, 
                src=Dungeon2Tiles.PLAYER, trg=Dungeon2Tiles.BAT),
            lambda: (0.0, jnp.zeros(env_map.shape, dtype=jnp.float32), jnp.full(2, dtype=jnp.int32, fill_value=-1))
        )

        stats = jnp.zeros(len(Dungeon2Metrics))
        stats = stats.at[Dungeon2Metrics.PATH_LENGTH].set(path_length)
        stats = stats.at[Dungeon2Metrics.N_ENEMIES].set(n_enemies)
        stats = stats.at[Dungeon2Metrics.N_KEYS].set(n_keys)
        stats = stats.at[Dungeon2Metrics.NEAREST_ENEMY].set(pe_path_length)
        stats = stats.at[Dungeon2Metrics.N_REGIONS].set(n_regions)
        stats = stats.at[Dungeon2Metrics.N_PLAYERS].set(n_players)
        stats = stats.at[Dungeon2Metrics.N_DOORS].set(n_doors)
        stats = stats.at[Dungeon2Metrics.N_TREASURES].set(n_treasures)
        stats = stats.at[Dungeon2Metrics.DOOR_ON_THRESHOLD].set(door_is_on_threshold)
        stats = stats.at[Dungeon2Metrics.TREASURE_IN_ROOM].set(treasure_in_room)
        state = Dungeon2State(
            stats=stats,
            player_key_flood_count=pk_flood_count,
            key_door_flood_count=kd_flood_count,
            player_enemy_flood_count=pe_flood_count,
            door_treasure_flood_count=dt_flood_count,
            enemy_xy=e_xy,
            k_xy=k_xy,
            d_xy=d_xy,
            t_xy=t_xy,
            ctrl_trgs=None,
        )
        return state

    def init_graphics(self):

        self.graphics = {
            Dungeon2Tiles.EMPTY: Image.open(
                f"{__location__}/tile_ims/empty.png"
            ).convert('RGBA'),
            Dungeon2Tiles.WALL: Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA'),
            Dungeon2Tiles.BORDER: Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA'),
            Dungeon2Tiles.KEY: Image.open(
                f"{__location__}/tile_ims/key.png"
            ).convert('RGBA'),
            Dungeon2Tiles.DOOR: Image.open(
                f"{__location__}/tile_ims/door.png"
            ).convert('RGBA'),
            Dungeon2Tiles.PLAYER: Image.open(
                f"{__location__}/tile_ims/player.png"
            ).convert('RGBA'),
            Dungeon2Tiles.BAT: Image.open(
                f"{__location__}/tile_ims/bat.png"
            ).convert('RGBA'),
            Dungeon2Tiles.SCORPION: Image.open(
                f"{__location__}/tile_ims/scorpion.png"
            ).convert('RGBA'),
            Dungeon2Tiles.SPIDER: Image.open(
                f"{__location__}/tile_ims/spider.png"
            ).convert('RGBA'),
            Dungeon2Tiles.TREASURE: Image.open(
                f"{__location__}/tile_ims/treasure.png"
            ).convert('RGBA'),
            len(Dungeon2Tiles): Image.open(f"{__location__}/tile_ims/path_g.png").convert(
                'RGBA'
            ),
            len(Dungeon2Tiles) + 1: Image.open(f"{__location__}/tile_ims/path_purple.png").convert(
                'RGBA'
            )
        }
        self.graphics = jnp.array(idx_dict_to_arr(self.graphics))
        super().init_graphics()

    def draw_path(self, lvl_img,env_map, border_size, path_coords_tpl, tile_size):
        assert len(path_coords_tpl) == 4
        lvl_img = draw_path(prob=self, lvl_img=lvl_img, env_map=env_map, border_size=border_size,
                            path_coords=path_coords_tpl[0], tile_size=tile_size,
                            im_idx=-2)
        lvl_img = draw_path(prob=self, lvl_img=lvl_img, env_map=env_map, border_size=border_size,
                            path_coords=path_coords_tpl[1], tile_size=tile_size,
                            im_idx=-2)
        lvl_img = draw_path(prob=self, lvl_img=lvl_img, env_map=env_map, border_size=border_size,
                            path_coords=path_coords_tpl[2], tile_size=tile_size,
                            im_idx=-2)
        lvl_img = draw_path(prob=self, lvl_img=lvl_img, env_map=env_map, border_size=border_size,
                            path_coords=path_coords_tpl[3], tile_size=tile_size)
        return lvl_img

