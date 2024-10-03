from enum import IntEnum
from functools import partial
import math
from typing import Optional, Tuple, Union

import chex
from flax import struct
from gymnax import EnvParams, EnvState
import jax
import jax.numpy as jnp
import numpy as np
import PIL

from gymnax.environments.environment import Environment as GymnaxEnvironment
from envs.env import Environment
from envs.pcgrl_env import PCGRLObs, gen_static_tiles
from envs.probs.binary import BinaryMetrics, BinaryProblem
from envs.probs.dungeon import DungeonProblem
from envs.probs.maze import MazeProblem
from envs.probs.maze_play import MazePlayProblem
from envs.probs.problem import Problem, ProblemState
from envs.reps.narrow import NarrowRepresentation
from envs.reps.player import PlayerRepresentation
from envs.reps.turtle import MultiTurtleRepresentation, TurtleRepresentation
from envs.reps.wide import WideRepresentation
from envs.reps.nca import NCARepresentation
from envs.reps.representation import Representation, RepresentationState
from envs.utils import Tiles
from sawtooth import triangle_wave


@struct.dataclass
class PCGRLEnvState:
    env_map: chex.Array
    static_map: chex.Array
    rep_state: RepresentationState
    prob_state: Optional[ProblemState] = None
    step_idx: int = 0
    reward: np.float32 = 0.0
    pct_changed: np.float32 = 0.0
    done: bool = False


@struct.dataclass
class PlayPCGRLEnvParams:
    map_shape: Tuple[int, int] = (16, 16)
    rf_shape: Tuple[int, int] = (31, 31)
    n_agents: int = 1
    max_board_scans: float = 2.0
    multiagent: bool = False

    
def gen_init_maze(rng, map_shape, tile_enum: Tiles):
    init_map = jnp.full(map_shape, tile_enum.WALL)
    # Place an empty tile at a random position
    rng, _ = jax.random.split(rng)
    y, x = jax.random.randint(rng, (2,), 0, map_shape)
    init_map = init_map.at[y, x].set(tile_enum.EMPTY)

    # Define a function that recognizes all places where there are two wall tiles in a row, adjacent to an empty tile
    def is_wall(rng, init_map, y, x):
        pass


class PlayPCGRLEnv(Environment):
    def __init__(self, env_params: PlayPCGRLEnvParams):
        map_shape, rf_shape, n_agents, max_board_scans = \
            env_params.map_shape, env_params.rf_shape, env_params.n_agents, env_params.max_board_scans

        self.map_shape = map_shape
        self.n_agents = n_agents
        self.multiagent = env_params.multiagent or self.n_agents > 1

        self.prob: Problem
        self.prob = MazePlayProblem(map_shape=map_shape, ctrl_metrics=[])

        self.tile_enum = self.prob.tile_enum
        self.tile_probs = self.prob.tile_probs
        rng = jax.random.PRNGKey(0)  # Dummy random key
        map_data = self.prob.gen_init_map(rng)
        env_map, actual_map_shape = map_data

        self.rep: Representation
        self.rep = PlayerRepresentation(env_map=env_map, rf_shape=rf_shape,
                                    tile_enum=self.tile_enum,
                                    act_shape=(1,1), map_shape=map_shape)

        self.max_steps = self.rep.max_steps
        self.tile_size = self.prob.tile_size

    def init_graphics(self):
        self.prob.init_graphics()

    @partial(jax.jit, static_argnums=(0, 2))
    def reset_env(self, rng, env_params: PlayPCGRLEnvParams) \
            -> Tuple[chex.Array, PCGRLEnvState]:
        map_data = self.prob.gen_init_map(rng)
        env_map, actual_map_shape = map_data
        frz_map = gen_static_tiles(rng, 0.0, 0, self.map_shape)

        rng, _ = jax.random.split(rng)
        rep_state = self.rep.reset(frz_map, rng)

        rng, _ = jax.random.split(rng)
        _, prob_state = self.prob.reset(env_map, rep_state.pos, rng)

        obs = self.get_obs(
            env_map=env_map, static_map=frz_map, rep_state=rep_state, prob_state=prob_state)

        env_state = PCGRLEnvState(env_map=env_map, static_map=frz_map,
                                  rep_state=rep_state, prob_state=prob_state,
                                  step_idx=0, done=False)

        return obs, env_state

    def get_obs(self, env_map, static_map, rep_state, prob_state):
        rep_obs = self.rep.get_obs(env_map, static_map, rep_state)
        prob_obs = self.prob.observe_ctrls(prob_state)
        obs = PCGRLObs(map_obs=rep_obs, flat_obs=prob_obs)
        return obs

    @partial(jax.jit, static_argnums=(0, 4))
    def step_env(self, rng, env_state: PCGRLEnvState, action, env_params):
        action = action[..., None]
        # if self.n_agents == 1:
        if not self.multiagent:
            action = action[0]
        env_map, map_changed, rep_state = self.rep.step(
            env_map=env_state.env_map, action=action,
            rep_state=env_state.rep_state, step_idx=env_state.step_idx
        )
        env_map = jnp.where(env_state.static_map == 1,
                            env_state.env_map, env_map,
        )
        n_tiles_changed = jnp.sum(jnp.where(env_map != env_state.env_map, 1, 0))
        pct_changed = n_tiles_changed / math.prod(self.map_shape)
        pct_changed = env_state.pct_changed + pct_changed
        reward, prob_state = jax.lax.cond(
            map_changed,
            lambda env_map: self.prob.step(env_map, env_state.prob_state, rep_state.pos),
            lambda _: (0., env_state.prob_state),
            env_map,
        )
        obs = self.get_obs(
            env_map=env_map, static_map=env_state.static_map,
            rep_state=rep_state, prob_state=prob_state)
        done = self.is_terminal(env_state, env_params)
        step_idx = env_state.step_idx + 1
        env_state = PCGRLEnvState(
            env_map=env_map, static_map=env_state.static_map,
            rep_state=rep_state, done=done, reward=reward,
            prob_state=prob_state, step_idx=step_idx, pct_changed=pct_changed,)

        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(env_state),
            reward,
            done,
            {"discount": self.discount(env_state, env_params)},
        )
    
    def is_terminal(self, state: PCGRLEnvState, params: PlayPCGRLEnvParams) \
            -> bool:
        """Check whether state is terminal."""
        done = state.step_idx >= (self.rep.max_steps - 1)
        # done = jnp.logical_or(done, jnp.logical_and(
        #     params.change_pct > 0, state.pct_changed >= params.change_pct))
        return done

    def render(self, env_state: PCGRLEnvState):
        # TODO: Refactor this into problem
        path_coords = self.prob.get_path_coords(
            env_map=env_state.env_map,
            prob_state=env_state.prob_state)
        return render_map(self, env_state, path_coords)

    @property
    def default_params(self) -> PlayPCGRLEnvParams:
        return PlayPCGRLEnvParams(map_shape=(16, 16))

    def action_space(self, env_params: PlayPCGRLEnvParams) -> int:
        return self.rep.action_space()

    def observation_space(self, env_params: PlayPCGRLEnvParams) -> int:
        return self.rep.observation_space()

    def action_shape(self):
        return (self.n_agents, *self.act_shape, len(self.tile_enum) - 1)

    def gen_dummy_obs(self, env_params: PlayPCGRLEnvParams):
        map_x = jnp.zeros((1,) + self.observation_space(env_params).shape)
        ctrl_x = jnp.zeros((1, 1))
        return PCGRLObs(map_x, ctrl_x)

    def sample_action(self, rng):
        action_shape = self.action_shape()
        # Sample an action from the action space
        n_dims = len(action_shape)
        act_window_shape = action_shape[:-1]
        n_tile_types = action_shape[-1]
        return jax.random.randint(rng, act_window_shape, 0, n_tile_types)[None, ...]


def render_map(env: PlayPCGRLEnv, env_state: PCGRLEnvState,
               path_coords: chex.Array):
    tile_size = env.prob.tile_size
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
            tile_img = env.prob.graphics[env_map[y, x]]
            lvl_img = lvl_img.at[y*tile_size: (y+1)*tile_size,
                                 x*tile_size: (x+1)*tile_size, :].set(tile_img)

    # Path, if applicable
    tile_img = env.prob.graphics[-1]

    def draw_path_tile(carry):
        path_coords, lvl_img, i = carry
        y, x = path_coords[i]
        tile_type = env_map[y + border_size[0]][x + border_size[1]]
        empty_tile = int(Tiles.EMPTY)
        lvl_img = jax.lax.cond(
            tile_type == empty_tile,
            lambda: jax.lax.dynamic_update_slice(lvl_img, tile_img,
                                               ((y + border_size[0]) * tile_size, (x + border_size[1]) * tile_size, 0)),
            lambda: lvl_img,)
                            
        return (path_coords, lvl_img, i+1)

    def cond(carry):
        path_coords, _, i = carry
        return jnp.all(path_coords[i] != jnp.array((-1, -1)))

    i = 0
    _, lvl_img, _ = jax.lax.while_loop(
        cond, draw_path_tile, (path_coords, lvl_img, i))

    clr = (255, 255, 255, 255)
    y_border = jnp.zeros((2, tile_size, 4), dtype=jnp.uint8)
    y_border = y_border.at[:, :, :].set(clr)
    x_border = jnp.zeros((tile_size, 2, 4), dtype=jnp.uint8)
    x_border = x_border.at[:, :, :].set(clr)
    if hasattr(env_state.rep_state, 'pos'):

        def render_pos(a_pos, lvl_img):
            y, x = a_pos
            y, x = y + border_size[0], x + border_size[1]
            y, x = y * tile_size, x * tile_size
            lvl_img = jax.lax.dynamic_update_slice(lvl_img, x_border, (y, x, 0))
            lvl_img = jax.lax.dynamic_update_slice(
                lvl_img, x_border, (y, x+tile_size-2, 0))
            lvl_img = jax.lax.dynamic_update_slice(lvl_img, y_border, (y, x, 0))
            lvl_img = jax.lax.dynamic_update_slice(
                lvl_img, y_border, (y+tile_size-2, x, 0))
            return lvl_img

        if env_state.rep_state.pos.ndim == 1:
            a_pos = env_state.rep_state.pos
            lvl_img = render_pos(a_pos, lvl_img)
        elif env_state.rep_state.pos.ndim == 2:
            for a_pos in env_state.rep_state.pos:
                lvl_img = render_pos(a_pos, lvl_img)

    clr = (255, 0, 0, 255)
    x_border = x_border.at[:, :, :].set(clr)
    y_border = y_border.at[:, :, :].set(clr)

    return lvl_img


def render_stats_jax(env: PlayPCGRLEnv, env_state: PCGRLEnvState, lvl_img: chex.Array):
    # TODO: jaxify this if possible. PROBLEM: can't get concrete values of stats

    tile_size = env.prob.tile_size
    env_map = env_state.env_map
    border_size = np.array((1, 1))
    env_map = jnp.pad(env_map, border_size, constant_values=Tiles.BORDER)
    full_height = len(env_map)
    full_width = len(env_map[0])

    row_height = 20
    n_rows = len(env_state.prob_state.stats) * 2 + 1
    lvl_img_text = jnp.zeros((row_height * n_rows, full_width*tile_size, 4), dtype=jnp.uint8)
    lvl_img_text = lvl_img_text.at[:].set((0, 0, 0, 255))

    metric_names = [m.name for m in BinaryMetrics]

    text_im_rows = []

    for i, s in enumerate(env_state.prob_state.stats):
        metric_name = metric_names[i]
        text = f'{metric_name}: {s}'
        char_arrs = [env.prob.ascii_chars_to_ims[c] for c in text]
        text_im = jnp.concatenate(char_arrs, axis=1)
        text_im = text_im.at[:, :, 3].set(255)
        text_im_rows.append(text_im)
        # lvl_img_text = lvl_img_text.at[2*i*row_height:(2*i+1)*row_height, :text_im.shape[1], :].set(text_im)

        trg = env_state.prob_state.ctrl_trgs[i]
        text = 'trg: ' + ''.join([' ' for _ in range(max(0, len(metric_name) + 2 - 5))]) + f'{trg}'
        char_arrs = [env.prob.ascii_chars_to_ims[c] for c in text]
        text_im = jnp.concatenate(char_arrs, axis=1)
        text_im = text_im.at[:, :, 3].set(255)
        text_im_rows.append(text_im)
        # lvl_img_text = lvl_img_text.at[(2*i+1)*row_height:(2*i+2)*row_height, :text_im.shape[1], :].set(text_im)

    text = f'{env.prob.observe_ctrls(env_state.prob_state)}'
    char_arrs = [env.prob.ascii_chars_to_ims[c] for c in text]
    text_im = jnp.concatenate(char_arrs, axis=1)
    text_im = text_im.at[:, :, 3].set(255)
    text_im_rows.append(text_im)

    max_row_width = max(max([r.shape[1] for r in text_im_rows]), lvl_img.shape[1])
    text_im_rows = [jnp.pad(r, ((0, 0), (0, max_row_width - r.shape[1]), (0, 0))) for r in text_im_rows]
    lvl_img_text = jnp.concatenate(text_im_rows, axis=0)

    lvl_img = jnp.pad(lvl_img, ((0, 0), (0, max(0, max_row_width - lvl_img.shape[1])), (0, 0)))

    lvl_img = jnp.concatenate((lvl_img, lvl_img_text), axis=0)
        

    return lvl_img


def render_stats(env: PlayPCGRLEnv, env_state: PCGRLEnvState, lvl_img: chex.Array):
    # TODO just use PIL draw functionality :)

    # jnp = np

    tile_size = env.prob.tile_size
    env_map = env_state.env_map
    border_size = np.array((1, 1))
    env_map = np.pad(env_map, border_size, constant_values=Tiles.BORDER)
    full_height = len(env_map)
    full_width = len(env_map[0])

    row_height = 20
    n_rows = len(env_state.prob_state.stats) * 2 + 1

    metric_names = [m.name for m in env.prob.metrics_enum]

    text_rows = []

    for i, s in enumerate(env_state.prob_state.stats):
        metric_name = metric_names[i]
        text = f'{metric_name}: {s}'
        text_rows.append(text)
        # lvl_img_text = lvl_img_text.at[2*i*row_height:(2*i+1)*row_height, :text_im.shape[1], :].set(text_im)

        trg = env_state.prob_state.ctrl_trgs[i]
        text = 'trg: ' + ''.join([' ' for _ in range(max(0, len(metric_name) + 2 - 5))]) + f'{trg}'
        text_rows.append(text)
        # lvl_img_text = lvl_img_text.at[(2*i+1)*row_height:(2*i+2)*row_height, :text_im.shape[1], :].set(text_im)

    text = f'obs: {env.prob.observe_ctrls(env_state.prob_state)}'
    text_rows.append(text)

    text = f'rew: {env_state.reward}'
    text_rows.append(text)

    max_text_chars = max([len(r) for r in text_rows])
    text = '\n'.join(text_rows)

    char_height, char_width = env.prob.render_font_shape
    total_width = max((max_text_chars * char_width, lvl_img.shape[1]))
    text_height = char_height * len(text_rows)
    lvl_img_text = PIL.Image.new('RGBA', (total_width, text_height), (0, 0, 0, 255))
    draw = PIL.ImageDraw.Draw(lvl_img_text)
    draw.text((0, 0), text, (255, 255, 255, 255))

    wid_padding = max(0, total_width - lvl_img.shape[1])
    lvl_img = np.pad(lvl_img, ((0, 0), (0, wid_padding), (0, 0)))
    lvl_img[:, :, 3] = 255

    lvl_img = np.concatenate((lvl_img, lvl_img_text), axis=0)
        

    return lvl_img