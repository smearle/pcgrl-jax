import math
import os
from typing import Tuple

import chex
from flax import struct
import flax
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from envs.pcgrl_env import Environment, PCGRLEnv, PCGRLEnvParams, render_map
from envs.probs.dungeon import DungeonProblem, DungeonTiles
from envs.reps.representation import get_ego_obs
from envs.utils import Tiles, idx_dict_to_arr


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), 'probs'))



@struct.dataclass
class PlayPCGRLEnvState:
    env_map: chex.Array
    last_action: int
    player_xy: chex.Array
    step_idx: int = 0

    # map_queue: chex.Array
    # map_i: int = 0


@struct.dataclass
class PlayPCGRLEnvParams:
    map_shape: chex.Array
    rf_shape: chex.Array
    # map_queue_size: int


@struct.dataclass
class PlayPCGRLObs:
    map_obs: chex.Array
    flat_obs: chex.Array


def gen_dummy_map(rng, map_shape):
    env_map = jnp.full(map_shape, DungeonTiles.WALL)
    env_map = env_map.at[8].set(DungeonTiles.EMPTY)
    n_tiles = math.prod(map_shape)
    # player_xy, door_xy = jax.random.choice(rng, n_tiles, (2,), replace=False)
    # player_xy, door_xy = jnp.unravel_index(jnp.array([player_xy, door_xy]), map_shape)
    player_xy = jnp.array([8, 0])
    door_xy = jnp.array([8, 15])
    env_map = env_map.at[tuple(player_xy)].set(DungeonTiles.PLAYER)
    env_map = env_map.at[tuple(door_xy)].set(DungeonTiles.DOOR)
    return env_map, player_xy


class PlayPCGRLEnv(Environment):
    player_dirs = jnp.array([
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
    ])

    tile_enum = DungeonTiles
    passable_tiles = jnp.array([DungeonTiles.EMPTY, DungeonTiles.KEY,
                                DungeonTiles.SCORPION, DungeonTiles.SPIDER,
                                DungeonTiles.BAT, DungeonTiles.DOOR])
    n_tile_types = len(tile_enum)

    def __init__(self, env_params):
        # super().__init__(env_params)
        self.map_shape = np.array(env_params.map_shape)
        self.max_steps = math.prod(self.map_shape)
        self.rf_shape = np.array(env_params.rf_shape)
        self.rf_off = int(max(np.ceil((self.rf_shape - 1) / 2)))

    def observation_space(self, env_params):
        return gymnax.environments.spaces.Box(
            low=0, high=1, shape=(*self.rf_shape, self.n_tile_types))
            # low=0, high=1, shape=(*self.map_shape, self.n_tile_types))

    def reset_env(self, rng, env_params: PCGRLEnvParams) \
            -> Tuple[chex.Array, PlayPCGRLEnvState]:
        # map_queue = jnp.full((env_params.map_queue_size, *self.map_shape), self.tile_enum.BORDER)
        env_map, player_xy = gen_dummy_map(rng, self.map_shape)
        state = PlayPCGRLEnvState(
            last_action=0,
            step_idx=0,
            env_map=env_map,
            # map_queue=None,
            player_xy=player_xy,
        )
        obs = self.get_obs(state)
        return obs, state

    def step_env(self, rng, state: PlayPCGRLEnvState, action, params: PlayPCGRLEnvParams):
        direction = self.player_dirs[action]
        env_map = state.env_map
        player_xy = state.player_xy + direction
        player_xy = np.clip(player_xy, 0, self.map_shape - 1)
        is_valid_move = jnp.any(env_map[tuple(player_xy)] == self.passable_tiles)

        # Update the player's position
        player_xy = jax.lax.select(
            is_valid_move,
            player_xy,
            state.player_xy)

        def move_player(env_map):
            env_map = env_map.at[tuple(state.player_xy)].set(self.tile_enum.EMPTY)
            env_map = env_map.at[tuple(player_xy)].set(self.tile_enum.PLAYER)
            return env_map

        # Determine if player will be on top of door and assign reward
        reached_goal = env_map[tuple(player_xy)] == self.tile_enum.DOOR
        reward = jax.lax.select(
            reached_goal,
            10.0,
            0.0)

        # Update map with player movement
        # env_map = jax.lax.cond(
        #     is_valid_move,
        #     lambda env_map: move_player(env_map),
        #     lambda env_map: env_map,
        #     state.env_map)
        
        env_map, player_xy = jax.lax.cond(
            reached_goal,
            lambda env_map, player_xy: gen_dummy_map(rng, self.map_shape),
            lambda env_map, player_xy: (env_map, player_xy),
            env_map, player_xy
        )

        # done = jnp.logical_or(reached_goal, (state.step_idx >= (self.max_steps - 1)))
        done = state.step_idx >= (self.max_steps - 1)
        state = PlayPCGRLEnvState(
            last_action=action,
            env_map=env_map,
            player_xy=player_xy,
            step_idx=state.step_idx + 1,) 
        obs = self.get_obs(state)
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": jax.lax.select(done, 0.0, 1.0)}
        )

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def get_obs(self, state: PlayPCGRLEnvState):
        padded_env_map = jnp.pad(
            state.env_map, self.rf_off, mode='constant',
            constant_values=self.tile_enum.BORDER)
        rf_map_obs = jax.lax.dynamic_slice(
            padded_env_map,
            state.player_xy,
            self.rf_shape,
        )
        rf_map_obs = jax.nn.one_hot(rf_map_obs, self.n_tile_types)
        action_one_hot = jax.nn.one_hot(
            state.last_action, self.num_actions
        ).squeeze()
        # time_rep = jax.lax.select(
        #     params.normalize_time, time_normalization(state.time), state.time
        # )
        # time_rep = jnp.array(state.step_idx, dtype=jnp.float32)[None]
        # flat_obs = jnp.hstack([action_one_hot, time_rep])
        flat_obs = action_one_hot
        obs = PlayPCGRLObs(
            map_obs=rf_map_obs.at[:].set(0.0),
            flat_obs=flat_obs.at[:].set(0.0), 
        )
        return obs

    def gen_dummy_obs(self, env_params: PCGRLEnvParams):
        map_x = jnp.zeros((1,) + self.observation_space(env_params).shape)
        flat_x = jnp.zeros((1, 4))  # dummy hack
        return PlayPCGRLObs(map_x, flat_x)

    tile_size = 16

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
        }
        self.graphics = jnp.array(idx_dict_to_arr(self.graphics))

    def render(self, env_state: PlayPCGRLEnvState):
        # TODO: Refactor this into problem

        tile_size = DungeonProblem.tile_size
        env_map = env_state.env_map
        border_size = np.array((1, 1))
        env_map = jnp.pad(env_map, border_size, constant_values=Tiles.BORDER)
        full_height = len(env_map)
        full_width = len(env_map[0])
        lvl_img = jnp.zeros(
            (full_height*tile_size, full_width*tile_size, 4), dtype=jnp.uint8)
        lvl_img = lvl_img.at[:].set((0, 0, 0, 255))

        # Map tiles
        for y in range(len(env_map)):
            for x in range(len(env_map[y])):
                tile_img = self.graphics[env_map[y, x]]
                lvl_img = lvl_img.at[y*tile_size: (y+1)*tile_size,
                                    x*tile_size: (x+1)*tile_size, :].set(tile_img)


        return lvl_img