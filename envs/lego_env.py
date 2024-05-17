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
    problem: int = ProbEnum.LEGO
    representation: int = RepEnum.LEGO_REARRANGE
    map_shape: Tuple[int, int, int] = (6, 6, 6)
    act_shape: Tuple[int] = (1,1)
    max_steps_multiple: int = 25
    #static_tile_prob: Optional[float] = 0.0
    #n_freezies: int = 0
    n_agents: int = 1
    n_blocks: int = 20
    #max_board_scans: float = 3.0
    ctrl_metrics: Tuple = (1,1)
    reward: Optional[chex.Array] = None
  
    #change_pct: float = -1.0


def get_prob_cls(problem: str):
    return PROB_CLASSES[problem]


class LegoEnv(Environment):
    prob: Problem
    def __init__(self, env_params: LegoEnvParams):
        map_shape, act_shape, problem, representation, n_agents, n_blocks, max_steps_multiple, reward = (
            env_params.map_shape, env_params.act_shape, env_params.problem,
            env_params.representation, env_params.n_agents, env_params.n_blocks, env_params.max_steps_multiple, env_params.reward)

        self.map_shape = map_shape
        self.act_shape = act_shape
        self.n_agents = n_agents
        self.static_tile_prob = 0.0
        self.n_freezies = 0.0
        
    

        prob_cls = PROB_CLASSES[problem]
        self.prob = prob_cls(map_shape=map_shape, ctrl_metrics=env_params.ctrl_metrics, n_blocks = n_blocks, reward = reward)
        

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
                max_steps_multiple = max_steps_multiple,
                reward = reward
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

        # frz_map = jnp.zeros(self.map_shape)
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
    def step_env(self, rng, env_state: LegoEnvState, action, env_params):
        action = action[..., None]
        target_reached = self.is_terminal(env_state, env_params)
        
        rng, subkey = jax.random.split(rng)
        if self.n_agents == 1:
            action = action[0]
        env_map, rng, rep_state = self.rep.step(
            env_map=env_state.env_map, action=action,
            rep_state=env_state.rep_state,
            step_idx = env_state.step_idx,
            rng = subkey
        )

        reward, prob_state = self.prob.step(env_map, state=env_state.prob_state, blocks=rep_state.blocks)

        env_state = LegoEnvState(
            env_map=env_map, static_map=env_state.static_map,
            rep_state=rep_state, done=False, reward=reward,
            prob_state=prob_state, step_idx=env_state.step_idx + 1, queued_state = env_state.queued_state)
        
        done = self.is_terminal(env_state, env_params) & target_reached
 
        env_state = env_state.replace(done=done)

        obs = self.get_obs(env_state.rep_state, env_state.prob_state)
        
        # self.render(env_state)

        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(env_state),
            reward,
            done,
            {"discount": self.discount(env_state, env_params), 
             "footprint": prob_state.stats[LegoMetrics.FOOTPRINT], 
             "avg_height": prob_state.stats[LegoMetrics.AVG_HEIGHT], 
             "ctr_dist": prob_state.stats[LegoMetrics.CENTER],
             "last_action": action[0][0][0],
             "stats": prob_state.stats,
             "done": done,
             "step": env_state.step_idx
             
             }
        )
    
    def is_terminal(self, env_state: LegoEnvState, env_params) \
            -> bool:
        """Check whether state is terminal."""
        max_reps_hit = env_state.step_idx >= (self.rep.max_steps - 1) 
        done = jnp.logical_or(max_reps_hit, env_state.prob_state.done)
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
        #flatmap = jnp.count_nonzero(env_state.env_map, 1)
        inds = jnp.arange(env_state.env_map.shape[1])[jnp.newaxis, :, jnp.newaxis]
        nonzeros = env_state.env_map != 0
        flatmap = jnp.argmax(nonzeros*inds, axis=1)
        return render_map(env_state, flatmap, tile_size = self.prob.tile_size)

    def get_blocks(self, env_state):
        return env_state.rep_state.blocks
    
    
    @property
    def default_params(self) -> LegoEnvParams:
        return LegoEnvParams(map_shape=(6, 6*3-2, 6))
    
    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        # TO DO: NOT HARD CODED
        return 6

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