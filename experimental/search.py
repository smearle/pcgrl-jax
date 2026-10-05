from timeit import default_timer as timer
import functools

import chex
import jax
from jax import numpy as jnp
import numpy as np

from conf.config import TrainConfig
from utils import gymnax_pcgrl_make, init_config
from envs.pcgrl_env import PCGRLEnvState


Reward = chex.Array

config = TrainConfig(
    representation='wide',
    max_board_scans=1.0,
    act_shape=(1, 1),
    n_agents=1,
)
config = init_config(config)
env, env_params = gymnax_pcgrl_make(config.env_name, config)

key = jax.random.PRNGKey(0)

n_actions = env.action_space(env_params).n

reset_key = jax.random.split(key, n_actions)
obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)

action = jnp.arange(n_actions, dtype=jnp.int32).reshape(-1)
action = jax.nn.one_hot(action, n_actions)[...,None,None].astype(jnp.int32)

# Hash the environment state
def hash(env_map: np.array):
    return hash(env_map.tobytes()) 

def get_state_fitness(state: PCGRLEnvState):
    return state.prob_state.diameter

    visited = {hash(state.env_map): (env_map, get_state_fitness(state))}

    for _ in range(100):
        key = jax.random.split(key)[0]

        # Actually no randomness in environment.
        key_step = jax.random.split(key, n_actions)

        obs, state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            key_step, state, action, env_params
        )
        breakpoint()