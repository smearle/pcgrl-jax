from typing import Optional, Tuple

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from gymnax.environments.environment import Environment
from envs.pathfinding import get_path_coords
from envs.probs.binary import BinaryProblem
from envs.probs.problem import Problem, ProblemState
from envs.reps.narrow import NarrowRepresentation
from envs.reps.wide import WideRepresentation
from envs.reps.nca import NCARepresentation
from envs.reps.representation import Representation, RepresentationState
from envs.utils import Tiles


@struct.dataclass
class PCGRLEnvState:
    env_map: chex.Array
    static_map: chex.Array
    rep_state: RepresentationState
    prob_state: Optional[ProblemState] = None
    step_idx: int = 0


@struct.dataclass
class PCGRLEnvParams:
    pass
    # map_shape: Tuple[int, int]


def gen_init_map(rng, tile_enum, map_shape, tile_probs):
    init_map = jax.random.choice(
        rng, len(tile_enum), shape=map_shape, p=tile_probs)
    return init_map


def gen_static_tiles(rng, static_tile_prob, n_freezies, map_shape):
    static_rng, rng = jax.random.split(rng)
    static_tiles = jax.random.bernoulli(
        static_rng, p=static_tile_prob, shape=map_shape)
    if n_freezies > 0:
        freezie_xys = jax.random.randint(rng, shape=(
            n_freezies, 2), minval=0, maxval=map_shape)
        freezie_dirs = jax.random.randint(
            rng, shape=(n_freezies,), minval=0, maxval=2)
        freezie_lens_empty = jnp.ones((n_freezies, 2), dtype=jnp.int32)
        freezie_lens = jax.random.randint(rng, shape=(
            n_freezies,), minval=0, maxval=max(map_shape))
        freezie_lens = freezie_lens_empty.at[jnp.arange(
            n_freezies), freezie_dirs].set(freezie_lens)
        for xy, len in zip(freezie_xys, freezie_lens):
            # static_tiles = static_tiles.at[xy[0]:xy[0]+len[0], xy[1]:xy[1]+len[1]].set(1)
            jax.lax.dynamic_update_slice(static_tiles, jnp.ones((len)), xy)

        # frz_xy = jax.random.randint(rng, shape=(n_freezies, 2), minval=0, maxval=map_shape)
        # frz_len = jax.random.randint(rng, shape=(n_freezies, 2), minval=1,
        #                                 # maxval=(map_shape[0] - frz_xy[:, 0], map_shape[1] - frz_xy[:, 1]))
        #                                 maxval=map_shape)
        # frz_len_1 = jnp.ones((n_freezies, 2), dtype=jnp.int32)
        # # frz_len_1 = np.ones((n_freezies, 2), dtype=jnp.int32)
        # frz_dirs = jax.random.randint(rng, shape=(n_freezies,), minval=0, maxval=2)
        # # frz_dirs = np.array(frz_dirs)
        # frz_len = frz_len_1.at[frz_dirs].set(frz_len[frz_dirs])
        # # frz_len[frz_dirs] = frz_len_1[frz_dirs]
        # for xy, len in zip(frz_xy, frz_len):
        #     static_tiles = jax.lax.dynamic_update_slice(static_tiles, jnp.ones(len), xy)
    return static_tiles


class PCGRLEnv(Environment):
    def __init__(self, problem: str, representation: str,
                 map_shape: Tuple[int, int], rf_shape: Tuple[int, int],
                 act_shape: Tuple[int, int],
                 static_tile_prob, n_freezies, env_params: PCGRLEnvParams):
        self.map_shape = map_shape
        self.static_tile_prob = np.float32(static_tile_prob)
        self.n_freezies = np.int32(n_freezies)

        self.prob: Problem
        if problem == 'binary':
            self.prob = BinaryProblem(map_shape=self.map_shape)
        else:
            raise Exception(f'Problem {problem} not implemented')

        self.tile_enum = self.prob.tile_enum
        self.tile_probs = self.prob.tile_probs
        rng = jax.random.PRNGKey(0)  # Dummy random key
        env_map = gen_init_map(rng, self.tile_enum, self.map_shape,
                               self.tile_probs)

        self.rep: Representation
        if representation == 'narrow':
            self.rep = NarrowRepresentation(env_map=env_map, rf_shape=rf_shape,
                                            tile_enum=self.tile_enum,
                                            act_shape=act_shape)
        elif representation == 'nca':
            self.rep = NCARepresentation(env_map=env_map, rf_shape=rf_shape,
                                         tile_enum=self.tile_enum,
                                         act_shape=act_shape)
        elif representation == 'wide':
            self.rep = WideRepresentation(env_map=env_map, rf_shape=rf_shape,
                                          tile_enum=self.tile_enum,
                                          act_shape=act_shape)
        else:
            raise Exception(f'Representation {representation} not implemented')

    def reset_env(self, rng, env_params: PCGRLEnvParams) \
            -> Tuple[chex.Array, PCGRLEnvState]:
        env_map = gen_init_map(rng, self.tile_enum,
                               self.map_shape, self.tile_probs)
        if self.static_tile_prob is not None or self.n_freezies > 0:
            static_map = gen_static_tiles(
                rng, self.static_tile_prob, self.n_freezies, self.map_shape)
        else:
            static_map = None

        rep_state = self.rep.reset(static_map)
        obs = self.rep.get_obs(
            env_map=env_map, static_map=static_map, rep_state=rep_state)

        _, prob_state = self.prob.get_stats(env_map)
        rep_state = self.rep.reset(static_map)
        env_state = PCGRLEnvState(env_map=env_map, static_map=static_map,
                                  rep_state=rep_state, prob_state=prob_state,
                                  step_idx=0)

        return obs, env_state

    def step_env(self, rng, env_state: PCGRLEnvState, action, env_params):
        env_map, map_changed, rep_state = self.rep.step(
            env_map=env_state.env_map, action=action,
            rep_state=env_state.rep_state, step_idx=env_state.step_idx)
        env_map = jnp.where(env_state.static_map == 1,
                            env_state.env_map, env_map)
        reward, prob_state = jax.lax.cond(
            map_changed,
            lambda env_map: self.prob.get_stats(env_map, env_state.prob_state),
            lambda _: (0., env_state.prob_state),
            env_map,
        )
        obs = self.rep.get_obs(
            env_map=env_map, static_map=env_state.static_map, rep_state=rep_state)
        done = self.is_terminal(env_state, env_params)
        step_idx = env_state.step_idx + 1
        env_state = PCGRLEnvState(env_map=env_map, static_map=env_state.static_map, rep_state=rep_state,
                                  prob_state=prob_state, step_idx=step_idx)

        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(env_state),
            reward,
            done,
            {"discount": self.discount(env_state, env_params)},
        )

    def is_terminal(self, state: PCGRLEnvState, params: PCGRLEnvParams) -> bool:
        """Check whether state is terminal."""
        done_steps = state.step_idx >= (self.rep.max_steps - 1)
        return done_steps

    def render(self, env_state: PCGRLEnvState):
        # TODO: Refactor this into problem
        path_coords = get_path_coords(
            env_state.prob_state.flood_path_state.flood_count, self.prob.max_path_len)
        return render_map(self, env_state, path_coords)

    @property
    def default_params(self) -> PCGRLEnvParams:
        return PCGRLEnvParams(map_shape=(16, 16))

    def action_space(self, env_params: PCGRLEnvParams) -> int:
        return self.rep.action_space()

    def observation_space(self, env_params: PCGRLEnvParams) -> int:
        return self.rep.observation_space()


tile_size = np.int8(16)


def render_map(env: PCGRLEnv, env_state: PCGRLEnvState, path_coords: chex.Array):
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
            tile_img = env.prob.graphics[env_map[y][x]]
            lvl_img = lvl_img.at[y*tile_size: (y+1)*tile_size,
                                 x*tile_size: (x+1)*tile_size, :].set(tile_img)

    # Path, if applicable
    tile_img = env.prob.graphics[-1]

    def draw_path_tile(carry):
        path_coords, lvl_img, i = carry
        y, x = path_coords[i]
        lvl_img = jax.lax.dynamic_update_slice(lvl_img, tile_img,
                                               ((y + border_size[0]) * tile_size, (x + border_size[1]) * tile_size, 0))
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
        y, x = env_state.rep_state.pos
        y, x = y + border_size[0], x + border_size[1]
        y, x = y * tile_size, x * tile_size
        lvl_img = jax.lax.dynamic_update_slice(lvl_img, x_border, (y, x, 0))
        lvl_img = jax.lax.dynamic_update_slice(
            lvl_img, x_border, (y, x+tile_size-2, 0))
        lvl_img = jax.lax.dynamic_update_slice(lvl_img, y_border, (y, x, 0))
        lvl_img = jax.lax.dynamic_update_slice(
            lvl_img, y_border, (y+tile_size-2, x, 0))

    clr = (255, 0, 0, 255)
    x_border = x_border.at[:, :, :].set(clr)
    y_border = y_border.at[:, :, :].set(clr)
    if env.static_tile_prob is not None or env.n_freezies > 0:
        static_map = env_state.static_map
        static_coords = jnp.argwhere(static_map,
                                     size=(
                                         env_map.shape[0]-border_size[0])*(env_map.shape[1]-border_size[1]),
                                     fill_value=-1)

        def draw_static_tile(carry):
            static_coords, lvl_img, i = carry
            y, x = static_coords[i]
            y, x = y + border_size[1], x + border_size[0]
            y, x = y * tile_size, x * tile_size
            lvl_img = jax.lax.dynamic_update_slice(
                lvl_img, x_border, (y, x, 0))
            lvl_img = jax.lax.dynamic_update_slice(
                lvl_img, x_border, (y, x+tile_size-2, 0))
            lvl_img = jax.lax.dynamic_update_slice(
                lvl_img, y_border, (y, x, 0))
            lvl_img = jax.lax.dynamic_update_slice(
                lvl_img, y_border, (y+tile_size-2, x, 0))
            return (static_coords, lvl_img, i+1)

        def cond(carry):
            static_coords, _, i = carry
            return jnp.all(static_coords[i] != jnp.array((-1, -1)))

        i = 0
        _, lvl_img, _ = jax.lax.while_loop(
            cond, draw_static_tile, (static_coords, lvl_img, i))

    return lvl_img
