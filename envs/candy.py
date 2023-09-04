from dataclasses import dataclass
from enum import IntEnum
from functools import partial
import os
import time
from timeit import default_timer as timer
from typing import Optional
import chex
import gymnax
from gymnax.environments import spaces
import jax
import jax.numpy as jnp
from jax import random
from flax import struct
import numpy as np
from PIL import Image
# import tkinter as tk


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


@struct.dataclass
class CandyState:
    # State of the candy game
    board: chex.Array
    rng: random.PRNGKey
    step_i: int = 0
    done: bool = False
    reward: int = 0

    
class CandyTiles(IntEnum):
    BLUE = 0
    RED = 1
    YELLOW = 2
    GREEN = 3
    EMPTY = 4


@struct.dataclass
class CandyParams:
    # Parameters for the candy game
    height: int = 8
    width: int = 8
    n_candy_types: int = 4
    cell_size: int = 50
    max_steps_in_episode: int = 500


DIRS = np.array([
    [0, 1], # right
    [1, 0], # down
    [0, -1], # left
    [-1, 0], # up
])
INVERSE_DIRS = {
    tuple(k): i for i, k in enumerate(DIRS)
}
DIRS = jnp.array(DIRS)


class Candy():
    tile_enum = CandyTiles
    def __init__(self, params: CandyParams):
        self.params = params
        self.height, self.width, self.n_candy_types = params.height, params.width, params.n_candy_types
        self.n_tile_types = len(self.tile_enum)
        self.n_dirs = 2  # Restrict player agent to right or down moves. Sufficient to cover all possible swaps.
        self.max_steps = params.max_steps_in_episode
        self.tile_size = 16
        self.map_shape = (self.height, self.width)

    def action_shape(self):
        return (self.height, self.width, self.n_dirs)

    def action_space(self, env_params) -> spaces.Discrete:
        return spaces.Discrete(self.height * self.width * self.n_dirs)

    def sample_action(self, rng):
        # Sample an action from the action space
        return random.randint(rng, (3,), np.zeros(3), np.array(self.action_shape()))

    def observation_shape(self):
        return (self.height, self.width, self.n_tile_types)

    def observation_space(self, env_params) -> spaces.Box:
        observation_shape = self.observation_shape()
        low = 1
        high = 1
        return spaces.Box(
            low, high, observation_shape, jnp.float32
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params: CandyParams, render=False):
        board = initialize_board(rng, np.int8(self.height), np.int8(self.width), np.int8(self.n_candy_types))
        if not render:
            board, rew, rng = simulate_board_jax(board, rng)
        if render:
            board, rew, frames, rng = simulate_board(board, rng, save_frames=True)
        obs = board
        state = CandyState(
            board=board,
            rng=rng,
            step_i=0,
            done=False,
            reward=0,
        )
        obs = self.get_observation(state)
        if render:
            return obs, state, frames
        return obs, state
    
    def render(self, state):
        return render_map(self, state)

    def get_observation(self, state: CandyState):
        # Get one-hot encoding of the board
        obs = jax.nn.one_hot(state.board, self.n_tile_types)
        return obs

    @partial(jax.jit, static_argnums=(0, 3))
    def step(self, key, state: CandyState, action, params, render=False):
        # Remove batch, player, x, and y dimensions
        action = action[0, 0, 0, 0]
        action = jnp.unravel_index(action, self.action_shape())
        x, y, direction = action
        coord1 = jnp.array([x, y])
        moved_board = move_element(state.board, coord1, direction)
        if not render:
            moved_board, reward, rng = simulate_board_jax(moved_board, state.rng)
        else:
            moved_board, reward, frames, rng = simulate_board(moved_board, state.rng, save_frames=True)
        board = jax.lax.cond(reward > 0, lambda: moved_board, lambda: state.board)
        state = CandyState(
            board=board,
            rng=rng,
            step_i=state.step_i + 1,
            done=state.step_i >= 500,
            reward=reward
        )
        obs = self.get_observation(state)
        done = state.done
        info = {}
        if render:
            info['frames'] = frames
        return obs, state, reward, done, info

    def step_render(self, action, state: CandyState):
        x, y, direction = action
        coord1 = jnp.array([x, y])
        moved_board = move_element(state.board, coord1, direction)
        all_frames = [state.board, moved_board]
        moved_board, reward, frames, rng = simulate_board(moved_board, state.rng, save_frames=True)
        all_frames.extend(frames)
        board = moved_board if reward > 0 else state.board
        state = CandyState(
            board=board,
            rng=rng,
            step_i=state.step_i + 1,
            done=state.step_i >= 500,
            reward=reward
        )
        obs = state.board
        done = state.done
        info = {}
        info['frames'] = all_frames
        return obs, state, reward, done, info


    def init_graphics(self):
        self.graphics = [0] * (len(self.tile_enum))
        self.graphics[CandyTiles.EMPTY] = Image.open(
                f"{__location__}/probs/tile_ims/empty.png"
            ).convert('RGBA')
        self.graphics[CandyTiles.BLUE] = Image.open(
                f"{__location__}/probs/tile_ims/player.png"
            ).convert('RGBA')
        self.graphics[CandyTiles.RED] = Image.open(
                f"{__location__}/probs/tile_ims/key.png"
            ).convert('RGBA')
        self.graphics[CandyTiles.GREEN] = Image.open(
                f"{__location__}/probs/tile_ims/spider.png"
            ).convert('RGBA')
        self.graphics[CandyTiles.YELLOW] = Image.open(
                f"{__location__}/probs/tile_ims/door.png"
            ).convert('RGBA')
        self.graphics = jnp.array(self.graphics)

def initialize_board(key, n, m, num_types):
    # Initialize board
    return random.randint(key, (n, m), 0, num_types)


def remove_matches(board):
    # Check for matches in the board and remove them
    hor_match = jnp.equal(board[:, :-2], board[:, 1:-1]) & jnp.equal(board[:, 1:-1], board[:, 2:])
    ver_match = jnp.equal(board[:-2, :], board[1:-1, :]) & jnp.equal(board[1:-1, :], board[2:, :])
    
    hor_match = jnp.pad(hor_match, ((0, 0), (1, 1)), mode='constant', constant_values=False)
    ver_match = jnp.pad(ver_match, ((1, 1), (0, 0)), mode='constant', constant_values=False)

    # Extend the match to include both end tiles for horizontal matches
    hor_match_ext = hor_match | jnp.pad(hor_match[:, 1:], ((0, 0), (0, 1)), mode='constant', constant_values=False) | \
                    jnp.pad(hor_match[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=False)
    hor_match_ext = jnp.where(board != -1, hor_match_ext, False)

    # Extend the match to include both end tiles for vertical matches
    ver_match_ext = ver_match | jnp.pad(ver_match[1:, :], ((0, 1), (0, 0)), mode='constant', constant_values=False) | \
                    jnp.pad(ver_match[:-1, :], ((1, 0), (0, 0)), mode='constant', constant_values=False)
    ver_match_ext = jnp.where(board != -1, ver_match_ext, False)


    matched_hor = jnp.any(hor_match_ext)
    full_match = jax.lax.cond(
        matched_hor,
        lambda: hor_match_ext,
        lambda: ver_match_ext,
    )
    reward = jnp.sum(full_match)

    new_board = jnp.where(full_match, -1, board)
    return new_board, reward


def apply_gravity(board):
    new_cols = [apply_gravity_col(board, i) for i in range(board.shape[1])]
    new_board = jnp.concatenate(new_cols, axis=1)
    return new_board


def apply_gravity_col(board, col):
    old_col = board[:, col]
    candy_idxs = jnp.argwhere(old_col[::-1] != -1, size=board.shape[0], fill_value=-1)
    ref_col = jnp.concatenate([old_col[::-1], jnp.array([-1])])
    new_col = ref_col[candy_idxs][::-1]
    return new_col


def refill_board(board, rng, num_types):
    # Refill the board
    rng = random.split(rng)[0]
    random_values = random.randint(rng, board.shape, 0, num_types)
    return jnp.where(board == -1, random_values, board), rng


def swap_elements(board, coord1, coord2):
    # jax.debug.print(f"Swapping {coord1} and {coord2}")
    # Swap two elements in the board
    coord1, coord2 = tuple(coord1), tuple(coord2)
    board = jnp.array(board)
    temp = board[coord1[0], coord1[1]]
    board = board.at[coord1[0], coord1[1]].set(board[coord2[0], coord2[1]])
    board = board.at[coord2[0], coord2[1]].set(temp)
    return board

def swap_elements_np(board, coord1, coord2):
    print(f"Swapping {coord1} and {coord2}")
    # Swap two elements in the board
    temp = board[coord1]
    board[coord1] = board[coord2]
    board[coord2] = temp
    return board


def move_element(board, coord1, dir):
    dir = DIRS[dir]
    # If the move is valid, move the element in the board
    coord2 = coord1 + dir
    # Clip in each dimension to avoid overhand
    coord2 = jnp.clip(coord2, jnp.array([0, 0]), jnp.array(board.shape) - 1)
    return swap_elements(board, coord1, coord2)


@jax.jit
def simulate_board_jax(board, rng):
    # jax.debug.print(f"simulating in jax:\n{board}")
    # Step through the board
    rew = 0
    old_board = board
    carry = step_board_all_while((old_board, board, 0, rng))
    carry = jax.lax.while_loop(
        lambda x: jnp.any(x[0] != x[1]),
        lambda x: step_board_all_while(x),
        carry,
    )
    _, board, rew, rng = carry
    return board, rew, rng

def simulate_board(board, rng, save_frames=True):
    old_board = board
    # _, board, reward, rng = step_board_all_while((old_board, board, 0, rng))
    board, reward, rng, boards = step_board_all_render(board, 0, rng)
    if save_frames:
        frames = boards
    else:
        frames = None
    i = 0
    while not jnp.all(old_board == board):
        old_board = board
        # _, board, reward, rng = step_board_all_while((old_board, board, reward, rng))
        board, reward, rng, boards = step_board_all_render(board, reward, rng)
        if save_frames:
            frames.extend(boards)
        i += 1
    return board, reward, frames, rng


def step_board_all_while(carry):
    _, board, rew, rng = carry
    old_board = board
    board, rew, rng = step_board_all(board, rew, rng)
    return (old_board, board, rew, rng)


def step_board_all(board, rew, rng):
    rew_old = rew
    board, rew = remove_matches(board)
    rew = rew + rew_old
    board = apply_gravity(board)
    board, rng = refill_board(board, rng, 4)
    return board, rew, rng, 

def step_board_all_render(board, rew, rng):
    rew_old = rew
    boards = []
    board, rew = remove_matches(board)
    boards.append(board)
    rew = rew + rew_old
    board = apply_gravity(board)
    boards.append(board)
    board, rng = refill_board(board, rng, 4)
    boards.append(board)
    return board, rew, rng, boards


def step_board_i(board, rng, cycle_i):
    if cycle_i == 0:
        board = remove_matches(board)
    elif cycle_i == 1:
        board = apply_gravity(board)
    elif cycle_i == 2:
        board, rng = refill_board(board, rng, 4)
    cycle_i = (cycle_i + 1) % 3
    return board, rng, cycle_i


def main():
    n_envs = 100
    env_params = CandyParams()
    seed = 0
    rng = random.PRNGKey(seed)
    rng_reset = random.split(rng, n_envs)
    env = Candy(env_params)
    obs, state = jax.vmap(env.reset, in_axes=(0, None), out_axes=0)(rng_reset, env_params)

    def step_env(key: random.PRNGKey, state: CandyState, action: chex.Array, params: CandyParams):
        # Step through the environment
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = env.step(
            action, state
        )
        obs_re, state_re = env.reset(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    def rand_step_env(carry, _):
        state: CandyState
        params: CandyParams
        rng, state, params = carry
        rng_action = random.split(rng, n_envs)
        action = jax.vmap(env.sample_action, in_axes=0, out_axes=0)(rng_action)

        obs, state, reward, done, info = jax.vmap(
            step_env, in_axes=(0, 0, 0, None), out_axes=0
        )(rng_action, state, action, params
        )
        return (rng, state, params), (obs, state, reward, done, info)

    rand_step_env_jit = jax.jit(rand_step_env)

    while True:
        n_frames = 1000
        start_time = timer()
        jax.lax.scan(
            lambda carry, _: rand_step_env_jit(carry, None),
            xs=None,
            init=(rng, state, env_params),
            length=n_frames,
        )
        end_time = timer()
        print(f'jax scan took {end_time - start_time} seconds\nframe rate: {n_frames * n_envs / (end_time - start_time)} fps')


def render_map(env: Candy, env_state: CandyState):
    tile_size = env.tile_size
    env_map = env_state.board
    border_size = np.array((1, 1))
    full_height = len(env_map)
    full_width = len(env_map[0])
    lvl_img = jnp.zeros(
        (full_height*tile_size, full_width*tile_size, 4), dtype=jnp.uint8)
    lvl_img = lvl_img.at[:].set((0, 0, 0, 255))

    # Map tiles
    for y in range(len(env_map)):
        for x in range(len(env_map[y])):
            tile_img = env.graphics[env_map[y][x]]
            lvl_img = lvl_img.at[y*tile_size: (y+1)*tile_size,
                                 x*tile_size: (x+1)*tile_size, :].set(tile_img)

    return lvl_img



if __name__ == '__main__':
    main()




