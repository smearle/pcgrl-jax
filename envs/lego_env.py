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
from envs.pcgrl_env import PCGRLObs
from envs.probs.lego import LegoProblem, LegoProblemState, LegoMetrics
#from envs.probs.binary import BinaryMetrics, BinaryProblem
#from envs.probs.dungeon import DungeonProblem
#from envs.probs.maze import MazeProblem
#from envs.probs.maze_play import MazePlayProblem
from envs.probs.problem import Problem#, #ProblemState
#from envs.reps.narrow import NarrowRepresentation
#from envs.reps.player import PlayerRepresentation
#rom envs.reps.turtle import MultiTurtleRepresentation, TurtleRepresentation
#from envs.reps.wide import WideRepresentation
#from envs.reps.nca import NCARepresentation
from envs.reps.representation import Representation, RepresentationState
from envs.reps.lego_rearrange import LegoRearrangeRepresentation, LegoRearrangeRepresentationState
from envs.utils import Tiles
#from sawtooth import triangle_wave


class ProbEnum(IntEnum):
    BINARY = 0
    MAZE = 1
    DUNGEON = 2
    MAZE_PLAY = 3
    LEGO = 4

PROB_CLASSES = {
    ProbEnum.LEGO: LegoProblem,
    #ProbEnum.BINARY: BinaryProblem,
    #ProbEnum.MAZE: MazeProblem,
    #ProbEnum.DUNGEON: DungeonProblem,
    #ProbEnum.MAZE_PLAY: MazePlayProblem,
}

class RepEnum(IntEnum):
    NARROW = 0
    TURTLE = 1
    WIDE = 2
    NCA = 3
    PLAYER = 4
    LEGO_REARRANGE = 5

@struct.dataclass
class QueuedState:
    has_queued_ctrl_trgs: bool = False
    ctrl_trgs: Optional[chex.Array] = None
    has_queued_frz_map: bool = False
    frz_map: Optional[chex.Array] = None

@struct.dataclass
class LegoEnvState:
    env_map: chex.Array
    rep_state: LegoRearrangeRepresentationState
    prob_state: Optional[LegoProblemState] = None
    static_map: Optional[chex.Array]=None
    step_idx: int = 0
    reward: np.float32 = 0.0
    done: bool = False
    queued_state: Optional[QueuedState] = None
    #key: Optional[jax.random.PRNGKey] = None

@struct.dataclass
class LegoEnvParams:
    rf_shape: Tuple[int, int, int]
    problem: int = ProbEnum.LEGO
    representation: int = RepEnum.LEGO_REARRANGE
    map_shape: Tuple[int, int, int] = (6, 6*3-2, 6)
    act_shape: Tuple[int] = (1,1)
    max_steps_multiple: int = 25
    #static_tile_prob: Optional[float] = 0.0
    #n_freezies: int = 0
    n_agents: int = 1
    n_blocks: int = 20
    #max_board_scans: float = 3.0
    ctrl_metrics: Tuple = (1,1)
    #change_pct: float = -1.0

"""
def gen_static_tiles(rng, static_tile_prob, n_freezies, map_shape):
    static_rng, rng = jax.random.split(rng)
    static_tiles = jax.random.bernoulli(
        static_rng, p=static_tile_prob, shape=map_shape).astype(bool)
    return jnp.zeros(map_shape)

"""
"""

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
"""

def get_prob_cls(problem: str):
    return PROB_CLASSES[problem]


class LegoEnv(Environment):
    prob: Problem
    def __init__(self, env_params: LegoEnvParams):
        map_shape, act_shape, problem, representation, n_agents, n_blocks, max_steps_multiple = (
            env_params.map_shape, env_params.act_shape, env_params.problem,
            env_params.representation, env_params.n_agents, env_params.n_blocks, env_params.max_steps_multiple)

        self.map_shape = map_shape
        self.act_shape = act_shape
        self.n_agents = n_agents
        self.static_tile_prob = 0.0
        self.n_freezies = 0.0
        
    

        prob_cls = PROB_CLASSES[problem]
        self.prob = prob_cls(map_shape=map_shape, ctrl_metrics=env_params.ctrl_metrics)
        

        rng = jax.random.PRNGKey(0)  # Dummy random key
        env_map = jnp.zeros(map_shape)

        self.rep: Representation
        self.rep = representation
        if representation == RepEnum.LEGO_REARRANGE:
            self.rep = LegoRearrangeRepresentation(
                tile_enum = self.prob.tile_enum,
                act_shape=act_shape,
                env_shape = map_shape,
                n_blocks = n_blocks,
                max_steps_multiple = max_steps_multiple
                #max_board_scans=env_params.max_board_scans,
            )
        else:
            raise Exception(f'Representation {representation} not implemented')

        self.max_steps = self.rep.max_steps
        self.tile_size = self.prob.tile_size
        env_map = env_map
        #self.render(env_map)
        

   
    #def init_graphics(self):
    #    self.prob.init_graphics()


    @partial(jax.jit, static_argnums=(0, 2))
    def reset_env(self, rng, env_params: LegoEnvParams, queued_state: QueuedState) \
            -> Tuple[chex.Array, LegoEnvState]:
        
        env_map = jnp.zeros(self.map_shape) #self.prob.gen_init_map(rng)    

        frz_map = jnp.zeros(self.map_shape)
        #frz_map = #jax.lax.select(
            #queued_state.has_queued_frz_map,
            #queued_state.frz_map,
            #gen_static_tiles(rng, self.static_tile_prob, self.n_freezies, self.map_shape),
        #)    

        #env_map = jnp.where(frz_map == 1, self.tile_enum.WALL, env_map)  
  
        rng, _ = jax.random.split(rng)
        rep_state = self.rep.reset(rng)
        #rep_state = self.rep.reset(rng, frz_map = None)
         

        rng, _ = jax.random.split(rng)
        _, prob_state = self.prob.reset(blocks=rep_state.blocks, env_map = env_map)        

        obs = self.get_obs(
            rep_state=rep_state, prob_state=prob_state)
        #obs = self.get_obs(
        #    env_map=env_map, rep_state=rep_state, prob_state=prob_state)

        env_state = LegoEnvState(env_map=env_map,
                                  rep_state=rep_state, prob_state=prob_state,
                                  step_idx=0, done=False, queued_state=queued_state)

        return obs, env_state

    def get_obs(self, rep_state, prob_state):
        rep_obs = self.rep.get_obs(rep_state)
        prob_obs = self.prob.observe_ctrls(prob_state)
        obs = PCGRLObs(map_obs=rep_obs, flat_obs=prob_obs)
        return obs
    
    def queue_frz_map(self, queued_state, frz_map: chex.Array):
        queued_state = queued_state.replace(frz_map=frz_map, has_queued_frz_map=True)
        return queued_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, env_state: LegoEnvState, action, env_params):
        action = action[..., None]
        
        rng, subkey = jax.random.split(rng)
        if self.n_agents == 1:
            action = action[0]
        env_map, rng, rep_state = self.rep.step(
            env_map=env_state.env_map, action=action,
            rep_state=env_state.rep_state,
            step_idx = env_state.step_idx,
            rng = subkey
        )
        #env_map = jnp.where(env_state.static_map == 1,
        #                    env_state.env_map, env_map,
        #) 
        
        #env_state = jax.lax.cond(env_state.prob_state == None, env_state.prob_state = LegoProblemState(reward = 0), env_state)
        reward, prob_state = self.prob.step(env_map, state=env_state.prob_state, blocks=rep_state.blocks)
        
        
        obs = self.get_obs(env_state.rep_state, env_state.prob_state)
            #env_map=env_map, static_map=env_state.static_map,
            #rep_state=rep_state, prob_state=prob_state)
         
        done = self.is_terminal(env_state, env_params)
        step_idx = env_state.step_idx + 1
        env_state = LegoEnvState(
            env_map=env_map, static_map=env_state.static_map,
            rep_state=rep_state, done=done, reward=reward + rep_state.punish_term,
            prob_state=prob_state, step_idx=step_idx, queued_state = env_state.queued_state)
        
        self.render(env_state)

        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(env_state),
            reward+rep_state.punish_term,
            done,
            {"discount": self.discount(env_state, env_params), 
             "footprint": prob_state.stats[LegoMetrics.FOOTPRINT], 
             "avg_height": prob_state.stats[LegoMetrics.AVG_HEIGHT], 
             "last_action": action[0][0][0],
             "stats": prob_state.stats,
             "rotation": rep_state.rotation
             }
        )
    
    def is_terminal(self, state: LegoEnvState, params: LegoEnvParams) \
            -> bool:
        """Check whether state is terminal."""
        done = state.step_idx >= (self.rep.max_steps - 1)
        #done = jnp.logical_or(done, jnp.logical_and(
        #    params.change_pct > 0, state.pct_changed >= params.change_pct))
        return done

    """
    def render(self, env_state: LegoEnvState):
        # TODO: Refactor this into problem
        path_coords_tpl = self.prob.get_path_coords(
            env_map=env_state.env_map,
            prob_state=env_state.prob_state)
        return render_map(self, env_state, path_coords_tpl)
    """

    def render(self, env_state):
        #TO DO: For real, the leocad/ldraw rendering
        flatmap = jnp.count_nonzero(env_state.env_map, 1)
        return render_map(env_state, flatmap, tile_size = self.prob.tile_size)

    def get_blocks(self, env_state):
        return env_state.rep_state.blocks
    
    
    @property
    def default_params(self) -> LegoEnvParams:
        return LegoEnvParams(map_shape=(16, 16))
    
    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        # TO DO: NOT HARD CODED
        return 5

    def action_space(self, env_params: LegoEnvParams) -> int:
        return self.rep.action_space()

    def observation_space(self, env_params: LegoEnvParams) -> int:
        return self.rep.observation_space()

    def action_shape(self):
        return (self.n_agents, *self.act_shape)

    def gen_dummy_obs(self, env_params: LegoEnvParams):
        map_x = jnp.zeros((1,) + self.observation_space(env_params).shape)
        ctrl_x = jnp.zeros((1, len(env_params.ctrl_metrics)))
        return PCGRLObs(map_x, ctrl_x)

    def sample_action(self, rng):
        action_shape = self.action_shape()
        # Sample an action from the action space
        n_dims = len(action_shape)
        #act_window_shape = action_shape[:-1]
        n_tile_types = action_shape[-1]
        return jax.random.randint(rng, action_shape, -1, 1)[None, ...]
 

def render_map(env_state: LegoEnvState, flatmap: chex.Array, tile_size: int):
        #tile_size = env_state.prob.tile_size
        border_size = np.array((1, 1))
        env_map = jnp.pad(flatmap, border_size, constant_values=0)
        full_height = len(env_map)
        full_width = len(env_map[0])
        lvl_img = jnp.zeros(
            (full_height*tile_size, full_width*tile_size, 4), dtype=jnp.uint8)
        lvl_img = lvl_img.at[:].set((255, 255, 255, 255))

        gradient = jnp.array([
                [0,0,0],
                [255,255,120],
                [102,255,153],
                #[86,245,161],
                [73,234,167],
                [66,223,171],
                [64,212,174],
                [68,200,174],
                [76,188,172],
                [85,177,167],
                [94,165,161],
                [102,153,153],
                [65,146,156],
                [0,138,163],
                [0,129,171],
                [0,118,180],
                [0,106,186],
                [0,92,188],
                [0,75,184],
                [64,52,173],
                [80,41,175],
                [102,0,153]
        ])

        # Map tiles
        for y in range(1, len(env_map)-1):
            for x in range(1, len(env_map[y]) - 1):
                tile_img = jnp.zeros((16,16,4))
                col = gradient[env_map][x,y]
                tile_img = tile_img.at[:,:,0].set(col[0])
                tile_img = tile_img.at[:,:,1].set(col[1])
                tile_img = tile_img.at[:,:,2].set(col[2])
                tile_img = tile_img.at[:,:,3].set(255)

                lvl_img = lvl_img.at[y*tile_size: (y+1)*tile_size,
                                    x*tile_size: (x+1)*tile_size, :].set(tile_img)

        clr = (255, 0, 0, 255)
        y_border = jnp.zeros((2, tile_size, 4), dtype=jnp.uint8)
        y_border = y_border.at[:, :, :].set(clr)
        x_border = jnp.zeros((tile_size, 2, 4), dtype=jnp.uint8)
        x_border = x_border.at[:, :, :].set(clr)
        #"""
        if hasattr(env_state.rep_state, 'curr_block'):

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

            curr_block_ind = (env_state.rep_state.curr_block)%len(env_state.rep_state.blocks)
            curr_block = (env_state.rep_state.blocks[curr_block_ind])

            a_pos = (curr_block[2], curr_block[0])
            lvl_img = render_pos(a_pos, lvl_img)
        
        #"""
        return lvl_img