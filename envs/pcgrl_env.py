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
from envs.probs.binary import BinaryMetrics, BinaryProblem
from envs.probs.dungeon import DungeonProblem
from envs.probs.dungeon2 import Dungeon2Problem
from envs.probs.maze import MazeProblem
from envs.probs.maze_play import MazePlayProblem
from envs.probs.problem import MapData, Problem, ProblemState
from envs.reps.narrow import NarrowRepresentation
from envs.reps.player import PlayerRepresentation
from envs.reps.turtle import MultiTurtleRepresentation, TurtleRepresentation
from envs.reps.wide import WideRepresentation
from envs.reps.nca import NCARepresentation
from envs.reps.representation import Representation, RepresentationState
from envs.utils import Tiles
from sawtooth import triangle_wave


class ProbEnum(IntEnum):
    BINARY = 0
    MAZE = 1
    DUNGEON = 2
    MAZE_PLAY = 3
    DUNGEON2 = 4

PROB_CLASSES = {
    ProbEnum.BINARY: BinaryProblem,
    ProbEnum.MAZE: MazeProblem,
    ProbEnum.DUNGEON: DungeonProblem,
    ProbEnum.DUNGEON2: Dungeon2Problem,
    ProbEnum.MAZE_PLAY: MazePlayProblem,
}

class RepEnum(IntEnum):
    NARROW = 0
    TURTLE = 1
    WIDE = 2
    NCA = 3
    PLAYER = 4


# FIXME: This is a hack to allow backward compatibility, reloading models from
#   before we added the option to queue maps for use at env reset.
@struct.dataclass
class OldQueuedState:
    has_queued_ctrl_trgs: bool = False
    ctrl_trgs: Optional[chex.Array] = None
    has_queued_frz_map: bool = False
    frz_map: Optional[chex.Array] = None
    # has_queued_map: Optional[bool] = False
    # map: Optional[chex.Array] = None

@struct.dataclass
class QueuedState:
    has_queued_ctrl_trgs: bool = False
    ctrl_trgs: Optional[chex.Array] = None
    has_queued_frz_map: bool = False
    frz_map: Optional[chex.Array] = None
    has_queued_map: bool = False
    map: Optional[chex.Array] = None
### END FIXME ##################################################################


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
    queued_state: Optional[QueuedState] = None


@struct.dataclass
class PCGRLObs:
    map_obs: chex.Array
    flat_obs: chex.Array


@struct.dataclass
class PCGRLEnvParams:
    multiagent: bool = False
    problem: int = ProbEnum.BINARY
    representation: int = RepEnum.NARROW
    map_shape: Tuple[int, int] = (16, 16)
    act_shape: Tuple[int, int] = (1, 1)
    rf_shape: Tuple[int, int] = (31, 31)
    static_tile_prob: Optional[float] = 0.0
    n_freezies: int = 0
    n_agents: int = 1
    max_board_scans: float = 3.0
    ctrl_metrics: Tuple = ()
    change_pct: float = -1.0
    randomize_map_shape: bool = False
    empty_start: bool = False
    pinpoints: bool = False


def gen_static_tiles(rng, static_tile_prob, n_freezies, map_shape):
    static_rng, rng = jax.random.split(rng)
    static_tiles = jax.random.bernoulli(
        static_rng, p=static_tile_prob, shape=map_shape).astype(bool)


    # if n_freezies > 0:
    def gen_freezies(rng):
        def gen_freezie(rng):
            # Randomly select row or column 
            rng, rng_ = jax.random.split(rng)

            height = map_shape[0]
            width = map_shape[1]
            
            rect = jnp.ones(map_shape, dtype=jnp.float16)
            row = rect[0]
            col = rect[1]

            locs = jax.random.uniform(rng_, shape=(2,))
            r_loc, c_loc = locs

            # FIXME: These guys are generally too big
            r_tri = triangle_wave(jnp.arange(map_shape[1]) / map_shape[1], 
                                   x_peak=r_loc, period=2)
            c_tri = triangle_wave(jnp.arange(map_shape[0]) / map_shape[0], 
                                   x_peak=c_loc, period=2)
            rc_tris = jnp.stack((r_tri, c_tri))
            maxval = jnp.max(rc_tris)
            minval = jnp.min(rc_tris)
            rc_cutoff = jax.random.uniform(rng_, shape=(2,), minval=minval*1.5, maxval=maxval)
            r_cut, c_cut = rc_cutoff
            r_tri = jnp.where(r_tri > r_cut, 1, 0)
            c_tri = jnp.where(c_tri > c_cut, 1, 0)

            rect = rect * r_tri * c_tri[..., None]
            rect = rect.astype(bool)

            return rect

        frz_keys = jax.random.split(rng, n_freezies)

        rects = jax.vmap(gen_freezie, in_axes=0)(frz_keys)

        rects = jnp.clip(rects.sum(0), 0, 1).astype(bool)
        # static_tiles = rects | static_tiles
        return rects

    static_tiles = jax.lax.cond(
        n_freezies > 0,
        lambda rng: static_tiles | gen_freezies(rng),
        lambda _: static_tiles,
        rng,
    ) 

    return static_tiles


def get_prob_cls(problem: str):
    return PROB_CLASSES[problem]


class PCGRLEnv(Environment):
    pinpoints = False

    def __init__(self, env_params: PCGRLEnvParams):
        map_shape, act_shape, rf_shape, problem, representation, static_tile_prob, n_freezies, n_agents = (
            env_params.map_shape, env_params.act_shape, env_params.rf_shape, env_params.problem,
            env_params.representation, env_params.static_tile_prob, env_params.n_freezies, env_params.n_agents)

        self.multiagent = env_params.multiagent
        self.map_shape = env_params.map_shape
        self.act_shape = env_params.act_shape
        self.static_tile_prob = np.float32(static_tile_prob)
        queued_frz_map = jnp.zeros(map_shape, dtype=bool)
        has_queued_frz_map = False
        self.n_freezies = np.int32(n_freezies)
        self.n_agents = n_agents
        self.randomize_map_shape = env_params.randomize_map_shape
        self.empty_start = env_params.empty_start
        self.pinpoints = env_params.pinpoints

        prob_cls = PROB_CLASSES[problem]
        self.prob: Problem = prob_cls(map_shape=map_shape, ctrl_metrics=env_params.ctrl_metrics,
                                      pinpoints=self.pinpoints)

        self.tile_enum = self.prob.tile_enum
        self.tile_probs = self.prob.tile_probs
        rng = jax.random.PRNGKey(0)  # Dummy random key
        map_data = self.prob.gen_init_map(rng, randomize_map_shape=self.randomize_map_shape, 
                                         empty_start=self.empty_start, pinpoints=self.pinpoints)
        env_map, actual_map_shape = map_data.env_map, map_data.actual_map_shape

        self.rep: Representation
        self.rep = representation
        if representation == RepEnum.NARROW:
            self.rep = NarrowRepresentation(env_map=env_map, rf_shape=rf_shape,
                                            tile_enum=self.tile_enum,
                                            act_shape=act_shape,
                                            max_board_scans=env_params.max_board_scans,
                                            pinpoints=self.pinpoints,
                                            tile_nums=self.prob.tile_nums,
            )
        elif representation == RepEnum.NCA:
            self.rep = NCARepresentation(env_map=env_map, rf_shape=rf_shape,
                                         tile_enum=self.tile_enum,
                                         act_shape=act_shape,
                                         max_board_scans=env_params.max_board_scans,
                                        pinpoints=self.pinpoints,
                                        tile_nums=self.prob.tile_nums,
            )
        elif representation == RepEnum.WIDE:
            self.rep = WideRepresentation(env_map=env_map, rf_shape=rf_shape,
                                          tile_enum=self.tile_enum,
                                          act_shape=act_shape,
                                          max_board_scans=env_params.max_board_scans,
                                          pinpoints=self.pinpoints,
                                          tile_nums=self.prob.tile_nums,
            )
        elif representation == RepEnum.TURTLE:
            # if n_agents > 1:
            if env_params.multiagent:
                self.rep = MultiTurtleRepresentation(
                    env_map=env_map, rf_shape=rf_shape,
                    tile_enum=self.tile_enum,
                    act_shape=act_shape,
                    map_shape=map_shape,
                    n_agents=n_agents,
                    max_board_scans=env_params.max_board_scans,
                    pinpoints=self.pinpoints,
                    tile_nums=self.prob.tile_nums,
                    )

            else:
                self.rep = TurtleRepresentation(env_map=env_map, rf_shape=rf_shape,
                                                tile_enum=self.tile_enum,
                                                act_shape=act_shape, map_shape=map_shape,
                                                pinpoints=self.pinpoints,
                                                tile_nums=self.prob.tile_nums,
                                                max_board_scans=env_params.max_board_scans,
                                                )
        elif representation == RepEnum.PLAYER:
            self.rep = PlayerRepresentation(env_map=env_map, rf_shape=rf_shape,
                                            tile_enum=self.tile_enum,
                                            act_shape=act_shape, map_shape=map_shape,
                                            pinpoints=self.pinpoints,
                                            tile_nums=self.prob.tile_nums,)
        else:
            raise Exception(f'Representation {representation} not implemented')

        self.max_steps = self.rep.max_steps
        self.tile_size = self.prob.tile_size

    # def set_ctrl_trgs(self, env_state, ctrl_trgs):
    #     # Assuming it's already batched
    #     env_state = PCGRLEnvState(
    #         env_map=env_state.env_map, static_map=env_state.static_map,
    #         rep_state=env_state.rep_state, done=env_state.done, reward=env_state.reward,
    #         prob_state=env_state.prob_state.replace(ctrl_trgs=ctrl_trgs), step_idx=env_state.step_idx)
    #     return env_state

    def init_graphics(self):
        self.prob.init_graphics()

    def queue_frz_map(self, queued_state, frz_map: chex.Array):
        queued_state = queued_state.replace(frz_map=frz_map, has_queued_frz_map=True)
        return queued_state

    @partial(jax.jit, static_argnames=('self',))
    def reset_env(self, rng, env_params: PCGRLEnvParams, queued_state: QueuedState) \
            -> Tuple[chex.Array, PCGRLEnvState]:
        queued_map_data = MapData(env_map=queued_state.map, actual_map_shape=jnp.array(self.map_shape))
        map_data = jax.lax.cond(
            queued_state.has_queued_map,
            lambda: queued_map_data,
            lambda: self.prob.gen_init_map(rng, randomize_map_shape=self.randomize_map_shape, 
                                   empty_start=self.empty_start, pinpoints=self.pinpoints),
        )
        env_map, actual_map_shape = map_data.env_map, map_data.actual_map_shape
        # frz_map = jax.lax.cond(
        #     self.static_tile_prob is not None or self.n_freezies > 0,
        #     lambda rng: gen_static_tiles(rng, self.static_tile_prob, self.n_freezies, self.map_shape),
        #     lambda _: None,
        #     rng,
        # )
        # frz_map = self.queued_frz_map if self.queued_frz_map is not None else gen_static_tiles(rng, self.static_tile_prob, self.n_freezies, self.map_shape)
        frz_map = jax.lax.select(
            queued_state.has_queued_frz_map,
            queued_state.frz_map,
            gen_static_tiles(rng, self.static_tile_prob, self.n_freezies, self.map_shape),
        )
        # Always freeze the border (in particular when using it to crop the map to some smaller size with
        # randomize_map_shape)
        frz_map = frz_map | jnp.where(env_map == Tiles.BORDER, True, False)

        if self.pinpoints:
            pinpoint_tiles = jnp.array([tile for tile, num in zip(self.tile_enum, self.prob.tile_nums) if num > 0])
            pinpoint_cells = jnp.isin(env_map, pinpoint_tiles)
            frz_map = jnp.where(pinpoint_cells, 1, frz_map)

        # env_map = jnp.where(frz_map == 1, self.tile_enum.WALL, env_map)  

        # if self.static_tile_prob is not None or self.n_freezies > 0:
        #     frz_map = gen_static_tiles(
        #         rng, self.static_tile_prob, self.n_freezies, self.map_shape)
        # else:
        #     frz_map = None

        rng, _ = jax.random.split(rng)
        rep_state = self.rep.reset(frz_map, rng)

        rng, _ = jax.random.split(rng)
        _, prob_state = self.prob.reset(env_map=env_map, rng=rng, queued_state=queued_state,
                                        actual_map_shape=actual_map_shape)

        obs = self.get_obs(
            env_map=env_map, frz_map=frz_map, rep_state=rep_state, prob_state=prob_state)

        env_state = PCGRLEnvState(env_map=env_map, static_map=frz_map,
                                  rep_state=rep_state, prob_state=prob_state,
                                  step_idx=0, done=False, queued_state=queued_state)

        return obs, env_state

    def get_obs(self, env_map, frz_map, rep_state, prob_state):
        rep_obs = self.rep.get_obs(env_map, frz_map, rep_state)
        prob_obs = self.prob.observe_ctrls(prob_state)
        obs = PCGRLObs(map_obs=rep_obs, flat_obs=prob_obs)
        return obs

    @partial(jax.jit, static_argnums=(0, 4))
    def step_env(self, rng, env_state: PCGRLEnvState, action, env_params, agent_id):
        action = action[..., None]
        # if self.n_agents == 1:
        if not self.multiagent:
            action = action[0]
        env_map, map_changed, rep_state = self.rep.step(
            env_map=env_state.env_map, action=action,
            rep_state=env_state.rep_state, step_idx=env_state.step_idx, agent_id=agent_id
        )
        env_map = jnp.where(env_state.static_map == 1,
                            env_state.env_map, env_map,
        )
        n_tiles_changed = jnp.sum(jnp.where(env_map != env_state.env_map, 1, 0))
        pct_changed = n_tiles_changed / math.prod(self.map_shape)
        pct_changed = env_state.pct_changed + pct_changed
        reward, prob_state = jax.lax.cond(
            map_changed,
            lambda env_map: self.prob.step(env_map, env_state.prob_state),
            lambda _: (0., env_state.prob_state),
            env_map,
        )
        obs = self.get_obs(
            env_map=env_map, frz_map=env_state.static_map,
            rep_state=rep_state, prob_state=prob_state)
        done = self.is_terminal(env_state, env_params)
        step_idx = env_state.step_idx + 1
        env_state = PCGRLEnvState(
            env_map=env_map, static_map=env_state.static_map,
            rep_state=rep_state, done=done, reward=reward,
            prob_state=prob_state, step_idx=step_idx, pct_changed=pct_changed, queued_state=env_state.queued_state)

        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(env_state),
            reward,
            done,
            {"discount": self.discount(env_state, env_params)},
        )
    
    def is_terminal(self, state: PCGRLEnvState, params: PCGRLEnvParams) \
            -> bool:
        """Check whether state is terminal."""
        done = state.step_idx >= (self.rep.max_steps - 1)
        done = jnp.logical_or(done, jnp.logical_and(
            params.change_pct > 0, state.pct_changed >= params.change_pct))
        return done

    def render(self, env_state: PCGRLEnvState):
        # TODO: Refactor this into problem
        path_coords_tpl = self.prob.get_path_coords(
            env_map=env_state.env_map,
            prob_state=env_state.prob_state)
        return render_map(self, env_state, path_coords_tpl)

    @property
    def default_params(self) -> PCGRLEnvParams:
        return PCGRLEnvParams(map_shape=(16, 16))

    @property
    def dummy_queued_state(self) -> QueuedState:
        return gen_dummy_queued_state(self)

    def action_space(self, env_params: PCGRLEnvParams) -> int:
        return self.rep.action_space

    def observation_space(self, env_params: PCGRLEnvParams) -> int:
        return self.rep.observation_space()

    def action_shape(self):
        return (self.n_agents, *self.act_shape, len(self.tile_enum) - 1)

    def gen_dummy_obs(self, env_params: PCGRLEnvParams):
        map_x = jnp.zeros((1,) + self.observation_space(env_params).shape)
        ctrl_x = jnp.zeros((1, len(env_params.ctrl_metrics)))
        return PCGRLObs(map_x, ctrl_x)

    def sample_action(self, rng):
        action_shape = self.action_shape()
        # Sample an action from the action space
        n_dims = len(action_shape)
        act_window_shape = action_shape[:-1]
        n_tile_types = action_shape[-1]
        return jax.random.randint(rng, act_window_shape, 0, n_tile_types)[None, ...]


def gen_dummy_queued_state(env):
    """ Generated a QueuedState object to be passed to environments whenever 
    they reset. Normally these are a bunch of empty/None values which indicate
    to the environment to generate its own random initial state as necessary.
    """
    queued_state = QueuedState(
        has_queued_ctrl_trgs=False,
        has_queued_frz_map=False,
        has_queued_map=False,
        ctrl_trgs=jnp.zeros(len(env.prob.stat_trgs)),
        frz_map=jnp.zeros(env.map_shape, dtype=bool),
        map=jnp.zeros(env.map_shape, dtype=jnp.int32),
    )
    return queued_state


# HACK for backward compat
def gen_dummy_queued_state_old(env):
    queued_state = OldQueuedState(
        has_queued_ctrl_trgs=False,
        has_queued_frz_map=False,
        ctrl_trgs=jnp.zeros(len(env.prob.stat_trgs)),
        frz_map=jnp.zeros(env.map_shape, dtype=bool),
    )
    return queued_state


def flatten_obs(obs: PCGRLObs) -> chex.Array:
    map_obs = jnp.reshape(obs.map_obs, (-1,))
    obs = jnp.concatenate((map_obs, obs.flat_obs))
    return obs


@partial(jax.jit, static_argnums=(0,))
def render_map(env: PCGRLEnv, env_state: PCGRLEnvState,
               path_coords_tpl: chex.Array):
    tile_size = int(env.prob.tile_size)
    env_map = env_state.env_map
    border_size = np.array((1, 1))
    env_map = jnp.pad(env_map, border_size, constant_values=Tiles.BORDER)
    full_height = int(len(env_map))
    full_width = int(len(env_map[0]))
    lvl_img = jnp.zeros(
        (full_height*tile_size, full_width*tile_size, 4), dtype=jnp.uint8)
    lvl_img = lvl_img.at[:].set((0, 0, 0, 255))

    # Map tiles
    for y in range(len(env_map)):
        for x in range(len(env_map[y])):
            tile_img = env.prob.graphics[env_map[y, x]]
            lvl_img = lvl_img.at[y*tile_size: (y+1)*tile_size,
                                 x*tile_size: (x+1)*tile_size, :].set(tile_img)

    lvl_img = env.prob.draw_path(lvl_img=lvl_img, env_map=env_map,
                                 path_coords_tpl=path_coords_tpl, border_size=border_size, tile_size=tile_size)

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
    # if env.static_tile_prob > 0 or env.n_freezies > 0 or env_state.queued_state.has_queued_frz_map:

    def render_frozen_tiles(lvl_img):
        static_map = env_state.static_map

        # Don't render the frozenness of BORDER tiles when using them to crop the map to some smaller size with
        # randomize_map_shape
        borderless_env_map = env_map[border_size[0]:-border_size[0], border_size[1]:-border_size[1]]
        static_map = jnp.where(borderless_env_map == Tiles.BORDER, 0, static_map)

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

    render_frozen_tiles(lvl_img)
    # lvl_img = jax.lax.cond(
    #     # env.static_tile_prob > 0 or env.n_freezies > 0 or env_state.queued_state.has_queued_frz_map,

    #     # The order matters here. If we have concrete_bool, traced_bool, then concrete_bool, there is an issue,
    #     # but concrete_bool, concrete_bool, traced_bool is fine. LMAO.
    #     env.static_tile_prob > 0 or env.n_freezies > 0 or env.pinpoints or env_state.queued_state.has_queued_frz_map,

    #     lambda lvl_img: render_frozen_tiles(lvl_img),
    #     lambda lvl_img: lvl_img,
    #     lvl_img
    # )

    return lvl_img


def render_stats_jax(env: PCGRLEnv, env_state: PCGRLEnvState, lvl_img: chex.Array):
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


def render_stats(env: PCGRLEnv, env_state: PCGRLEnvState, lvl_img: chex.Array):
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