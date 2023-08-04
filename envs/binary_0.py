import os
from enum import Enum, IntEnum
import math
import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Iterable, Sequence, Tuple, Optional
import chex
from flax import struct
from flax.core.frozen_dict import unfreeze
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from PIL import Image
import numpy as np


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class Tiles(IntEnum):
    BORDER = 0
    EMPTY = 1
    WALL = 2
    GOAL = 3


@struct.dataclass
class FloodState:
    flood_input: chex.Array
    flood_count: chex.Array


@struct.dataclass
class EnvState:
    last_action: int
    last_reward: float
    pos: chex.Array
    env_map: chex.Array
    flood_state: FloodState
    path_length: int
    goal: chex.Array
    time: float


@struct.dataclass
class EnvParams:
    # github.com/uber-research/backpropamine/blob/180c9101fa5be5a2da205da3399a92773d395091/simplemaze/maze.py#L414-L431
    reward: float = 10.0 
    punishment: float = 0.0
    normalize_time: bool = False
    max_steps_in_episode: int = 200


class Flood(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(1, kernel_size=(3, 3), padding='SAME', kernel_init=constant(0.0), bias_init=constant(0.0))(x)
        return x


def get_path_coords(flood_count: chex.Array):
    """Get the path coordinates from the flood count."""
    dirs = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    # Get the coordinates of a tile where the count is 1
    xs, ys = jnp.unravel_index(jnp.argmin(jnp.where(flood_count == 0, 99999, flood_count)), flood_count.shape)
    path_coords = [jnp.array([xs, ys])]
    curr_val = flood_count[xs, ys]
    max_val = jnp.max(flood_count)
    while curr_val < max_val:
        # Get the coordinates of a neighbor tile where the count is curr_val + 1
        di = 0
        while di < len(dirs):
            d = dirs[di]
            d_coords = path_coords[-1] + d
            if curr_val + 1 == int(flood_count[d_coords[0], d_coords[1]]):
                path_coords.append(d_coords)
                curr_val += 1
                break 
            di += 1
        # raise Exception('Path died!')
    return path_coords


def gen_init_map(rng: chex.PRNGKey, maze_size: int, rf_size: int, tile_probs: Iterable[int]) -> chex.Array:
    """Randomly generate an initial level."""
    rand_maze = jax.random.choice(rng, len(tile_probs), shape=(maze_size, maze_size), p=tile_probs)
    # Randomly place the goal tile
    goal_pos = jax.random.randint(rng, (2,), 0, maze_size)
    rand_maze = rand_maze.at[goal_pos[0], goal_pos[1]].set(Tiles.GOAL)
    goal_pos += rf_size // 2
    # Need to add wall offset if receptive field size is large
    rf_offset = int((rf_size - 1) / 2)
    maze = jnp.pad(rand_maze, rf_offset, mode='constant', constant_values=Tiles.BORDER)
    return goal_pos, maze


# def reset_goal(
#     rng: chex.PRNGKey, available_goals: chex.Array, env_map: chex.Array, params: EnvParams
# ) -> chex.Array:
#     """Reset the goal state/position in the environment."""
#     goal_index = jax.random.randint(rng, (), 0, available_goals.shape[0])
#     goal = available_goals[goal_index][:]
#     env_map = env_map.at[goal[0], goal[1]].set(Tiles.GOAL)
#     return goal, env_map
#     # goal = available_goals[0][:]
#     # env_map = env_map.at[goal[0], goal[1]].set(Tiles.GOAL)
#     # return goal, env_map


def reset_pos(rng: chex.PRNGKey, coords: chex.Array) -> chex.Array:
    """Reset the position of the agent."""
    # pos_index = jax.random.randint(rng, (), 0, coords.shape[0])
    # return coords[pos_index][:]
    return coords[0][:]


def time_normalization(
    t: float, min_lim: float = -1.0, max_lim: float = 1.0, t_max: int = 100
) -> float:
    """Normalize time integer into range given max time."""
    return (max_lim - min_lim) * t / t_max + min_lim


class Binary0(environment.Environment):
    """
    JAX Compatible version of meta-maze environment (Miconi et al., 2019).
    Source: Comparable to
    github.com/uber-research/backpropamine/blob/master/simplemaze/maze.py
    """
    num_tiles = len(Tiles)
    tile_probs = [0.0] * num_tiles
    tile_probs[Tiles.EMPTY] = 0.5
    tile_probs[Tiles.WALL] = 0.5
    tile_probs = jnp.array(tile_probs)
    _graphics = {
        Tiles.EMPTY: Image.open(__location__ + "/probs/tile_ims/empty.png").convert('RGBA'),
        Tiles.WALL: Image.open(__location__ + "/probs/tile_ims/solid.png").convert('RGBA'),
        Tiles.BORDER: Image.open(__location__ + "/probs/tile_ims/solid.png").convert('RGBA'),
        Tiles.GOAL: Image.open(__location__ + "/probs/tile_ims/player.png").convert('RGBA'),
        "path": Image.open(__location__ + "/probs/tile_ims/path_g.png").convert('RGBA'),
    }

    def __init__(self, maze_size: int = 16, rf_size: int = 31):
        super().__init__()
        # Maze size and receptive field have to be uneven (centering)
        # assert maze_size % 2 != 0
        assert rf_size % 2 != 0 and rf_size > 1
        self.maze_size = maze_size
        self.rf_size = rf_size
        # Offset of walls top/bottom and left/right
        self.rf_off = jnp.int32((self.rf_size - 1) / 2)
        # Generate the maze layout
        rng = jax.random.PRNGKey(0) # This key doesn't matter since we'll reset before playing anyway(?)
        goal, self.env_map = gen_init_map(rng, maze_size, rf_size, self.tile_probs)
        # center = jnp.int32((self.env_map.shape[0] - 1) / 2 + self.rf_off - 1)
        center = jnp.int32((self.env_map.shape[0] - 1) / 2)
        self.center_position = jnp.array([center, center])
        # self.occupied_map = 1 - self.env_map
        self.agent_map_shape = self.env_map.shape[0] - 2 * self.rf_off, self.env_map.shape[1] - 2 * self.rf_off
        self.agent_coords = jnp.argwhere(self.env_map != Tiles.BORDER, size=math.prod(self.agent_map_shape))
        # self.get_available_goals()

        self.builds = jnp.array([Tiles.EMPTY, Tiles.WALL, 0, 0, 0, 0])
        self.directions = jnp.array([[0, 0], [0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])

        self.flood_net = Flood()
        init_x = jnp.zeros(self.agent_map_shape + (2,), dtype=jnp.float32)
        self.flood_params = unfreeze(self.flood_net.init(rng, init_x))
        flood_kernel = self.flood_params['params']['Conv_0']['kernel']
        # Walls on center tile prevent it from being flooded
        flood_kernel = flood_kernel.at[1, 1, 0].set(-5)
        # Flood at adjacent tile produces flood toward center tile
        flood_kernel = flood_kernel.at[1, 2, 1].set(1)
        flood_kernel = flood_kernel.at[2, 1, 1].set(1)
        flood_kernel = flood_kernel.at[1, 0, 1].set(1) 
        flood_kernel = flood_kernel.at[0, 1, 1].set(1)
        flood_kernel = flood_kernel.at[1, 1, 1].set(1)
        self.flood_params['params']['Conv_0']['kernel'] = flood_kernel
        # Determine max path length (within borders) to find upper bound on how long it may take to flood.
        self.max_path_length = math.ceil(math.prod(self.agent_map_shape) / 2) + max(self.agent_map_shape)

    def flood_step(self, flood_state: FloodState, unused):
        flood_input, flood_count = flood_state.flood_input, flood_state.flood_count
        occupied_map = flood_input[..., 0]
        flood_out = self.flood_net.apply(self.flood_params, flood_input)
        flood_out = jnp.clip(flood_out, a_min=0, a_max=1)
        flood_out = jnp.stack([occupied_map, flood_out[..., -1]], axis=-1)
        flood_count = flood_out[..., -1] + flood_count
        flood_state = FloodState(flood_input=flood_out, flood_count=flood_count)
        return flood_state, None

    # def get_available_goals(self):
    #     # self.traversible_coords = jnp.argwhere(self.env_map != Tiles.WALL, size=math.prod(self.agent_map_shape), fill_value=-1)
    #     # traversible_coords = []
    #     # Get all walkable positions or positions that can be goals
    #     # for y in range(self.env_map.shape[0]):
    #     #     for x in range(self.env_map.shape[1]):
    #     #         if self.env_map[y, x] == Tile.EMPTY:
    #     #             traversible_coords.append([y, x])
    #     # self.traversible_coords = jnp.array(traversible_coords)

    #     # Any open space in the map can be a goal for the agent
    #     # self.available_goals = self.traversible_coords
    #     self.available_goals = self.agent_coords

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        p = state.pos + self.directions[action]
        in_map = state.env_map[p[0], p[1]] != Tiles.BORDER
        new_pos = jax.lax.select(in_map, p, state.pos)
        # goal_reached = jnp.logical_and(
        #     new_pos[0] == state.goal[0], new_pos[1] == state.goal[1]
        # )
        # reward = (
        #     goal_reached * params.reward  # Add goal reward
        #     # + (1 - in_map) * params.punishment  # Add punishment for wall
        # )

        # Sample a new starting position for case when goal is reached
        # pos_sampled = reset_pos(key, self.coords)
        # new_pos = jax.lax.select(goal_reached, pos_sampled, new_pos)
        b = self.builds[action]  # Meaningless if agent is moving.
        nem = state.env_map.at[new_pos[0], new_pos[1]].set(jnp.array(b, int))
        # If agent isn't moving, then let it build
        valid_build = jnp.logical_and(b != 0, state.env_map[new_pos[0], new_pos[1]] != Tiles.GOAL)
        new_env_map = jax.lax.select(valid_build, nem, state.env_map)
        # Update state dict and evaluate termination conditions
        # reward = new_occ_map.sum() - state.occupied_map.sum()
        # new_path_length, flood_state = self.calc_path(state, new_occ_map)
        new_path_length, flood_state = jax.lax.cond(b != 0, lambda : self.calc_path(state.goal, new_env_map), lambda : (state.path_length, state.flood_state))
        last_path_length = state.path_length
        reward = new_path_length - last_path_length
        state = EnvState(action, reward, new_pos, new_env_map, flood_state, new_path_length, state.goal, state.time + 1)
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state, params)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def calc_path(self, goal, new_env_map: jnp.ndarray):
        new_env_map = jax.lax.dynamic_slice(
            new_env_map, 
            (self.rf_off, self.rf_off), 
            (self.agent_map_shape[0], self.agent_map_shape[1])
        )
        occupied_map = jnp.logical_or(new_env_map == Tiles.WALL, new_env_map == Tiles.BORDER).astype(jnp.float32)
        # init_flood = jnp.zeros(self.agent_map_shape, dtype=jnp.float32)
        # init_flood = init_flood.at[goal[0], goal[1]].set(1)
        init_flood = new_env_map == Tiles.GOAL
        init_flood_count = init_flood.copy()
        # Concatenate init_flood with new_occ_map
        flood_input = jnp.stack([occupied_map, init_flood], axis=-1)
        flood_state = FloodState(flood_input=flood_input, flood_count=init_flood_count)
        flood_state, _ = jax.lax.scan(self.flood_step, flood_state, None, self.max_path_length)
        path_length = jnp.clip(flood_state.flood_count.max() - jnp.where(flood_state.flood_count == 0, 99999, flood_state.flood_count).min(), 0)
        return path_length, flood_state

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        # Reset both the agents position and the goal location
        goal, env_map = gen_init_map(key, self.maze_size, self.rf_size, self.tile_probs)
        # occupied_map = (self.env_map == Tile.WALL)[self.rf_off:-self.rf_off, self.rf_off:-self.rf_off].astype(float)
        # self.get_available_goals()
        # goal, env_map = reset_goal(key, self.available_goals, env_map, params)
        path_length, flood_state = self.calc_path(goal, env_map)
        state = EnvState(0, 0.0, self.center_position, env_map, flood_state, path_length, goal, 0.0)
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Return observation from raw state trafo."""
        rf_obs = jax.lax.dynamic_slice(
            state.env_map,
            (state.pos[0] - self.rf_off, state.pos[1] - self.rf_off),
            (self.rf_size, self.rf_size),
        )
        # Convert to one-hot encoding
        rf_obs = jax.nn.one_hot(rf_obs, self.num_tiles).reshape(-1)
        action_one_hot = jax.nn.one_hot(
            state.last_action, self.num_actions
        ).squeeze()
        time_rep = jax.lax.select(
            params.normalize_time, time_normalization(state.time), state.time
        )
        return jnp.hstack([rf_obs, action_one_hot, state.last_reward, time_rep])

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        # done_steps = state.time >= params.max_steps_in_episode
        done_steps = state.time >= self.maze_size ** 2 * 2
        # Check if agent has found the goal
        # done_goal = jnp.logical_and(
        #     state.pos[0] == state.goal[0],
        #     state.pos[1] == state.goal[1],
        # )
        # done = jnp.logical_or(done_goal, done_steps)
        return done_steps

    def render(self, state: EnvState, params: EnvParams):
        return render_map(self, state, get_path_coords(state.flood_state.flood_count))

    @property
    def name(self) -> str:
        """Environment name."""
        return "Binary"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(6)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array(
            self.rf_size ** 2 * self.num_tiles * [0] + self.num_actions * [0] + [0, 0],
            dtype=jnp.float32,
        )
        high = jnp.array(
            self.rf_size ** 2 * self.num_tiles * [1]
            + self.num_actions * [1]
            + [1, params.max_steps_in_episode],
            dtype=jnp.float32,
        )
        return spaces.Box(
            low, high, (self.rf_size ** 2 * self.num_tiles + self.num_actions + 2,), jnp.float32
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "last_action": spaces.Discrete(self.num_actions),
                "last_reward": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "pos": spaces.Box(
                    jnp.min(self.traversible_coords),
                    jnp.max(self.traversible_coords),
                    (2,),
                    jnp.float32,
                ),
                "occupied_map": spaces.Box(
                    0, 1, (self.size, self.size), jnp.float32
                ),
                "goal": spaces.Box(
                    jnp.min(self.traversible_coords),
                    jnp.max(self.traversible_coords),
                    (2,),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )



tile_size = 16
def render_map(env: Binary0, state: EnvState, path_coords):
    map = np.array(state.env_map)
    border_size = (1,1)
    map = map[env.rf_off-border_size[0]:-env.rf_off+border_size[0], env.rf_off-border_size[1]:-env.rf_off+border_size[1]]
    full_width = len(map[0])
    full_height = len(map)
    lvl_image = Image.new("RGBA", (full_width*tile_size, full_height*tile_size), (0,0,0,255))

    # Background floor everywhere
    # for y in range(full_height):
    #     for x in range(full_width):
    #         lvl_image.paste(env._graphics['empty'], (x*tile_size, y*tile_size, (x+1)*tile_size, (y+1)*tile_size))

    # Borders
    # for y in range(full_height):
    #     for x in range(env.rf_off):
    #         lvl_image.paste(env._graphics[Tiles.BORDER], (x*tile_size, y*tile_size, (x+1)*tile_size, (y+1)*tile_size))
    #         lvl_image.paste(env._graphics[Tiles.BORDER], ((full_width-x-1)*tile_size, y*tile_size, (full_width-x)*tile_size, (y+1)*tile_size))
    # for x in range(full_width):
    #     for y in range(env.rf_off):
    #         lvl_image.paste(env._graphics[Tiles.BORDER], (x*tile_size, y*tile_size, (x+1)*tile_size, (y+1)*tile_size))
    #         lvl_image.paste(env._graphics[Tiles.BORDER], (x*tile_size, (full_height-y-1)*tile_size, (x+1)*tile_size, (full_height-y)*tile_size))

    # Map tiles
    for y in range(len(map)):
        for x in range(len(map[y])):
            tile_image = env._graphics[map[y][x]]
            lvl_image.paste(env._graphics[map[y][x]], (x*tile_size, y*tile_size, (x+1)*tile_size, (y+1)*tile_size), mask=tile_image)

    # Path, if applicable
    tile_graphics = env._graphics["path"]
    for (y, x) in path_coords:
        lvl_image.paste(tile_graphics, ((x + border_size[0]) * tile_size, (y + border_size[1]) * tile_size, (x+border_size[0]+1) * tile_size, (y+border_size[1]+1) * tile_size), mask=tile_graphics)

    y, x = state.pos
    # y -= env.rf_off-border_size[0]
    # x -= env.rf_off-border_size[1]
    im_arr = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
    clr = (255, 255, 255, 255)
    im_arr[(0, 1, -1, -2), :, :] = im_arr[:, (0, 1, -1, -2), :] = clr
    x_graphics = Image.fromarray(im_arr)
    lvl_image.paste(x_graphics, (x*tile_size, y*tile_size,
                                    (x+1)*tile_size,(y+1)*tile_size), x_graphics)
    return lvl_image