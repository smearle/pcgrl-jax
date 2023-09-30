'''
get the fitness of the evolved frz map (or other thingys we want to evolve)
'''
import os
from envs.utils import Tiles
from flax import struct
from typing import Optional
import chex
import jax
from jax import numpy as jnp
import numpy as np
from config import TrainConfig
from envs.pcgrl_env import PCGRLEnv, QueuedState
from tensorboardX import SummaryWriter

from utils import get_exp_dir

@struct.dataclass # need to make a carrier for for the fitness to the tensorboard logging? hmm unnecessary
class EvoState:
    top_fitness: Optional[chex.Array] = None
    frz_map: Optional[chex.Array] = None

def apply_evo(rng, frz_maps, env: PCGRLEnv, env_params, network_params, network,
              config: TrainConfig, runner_state):
    '''
    copy and mutate the frz maps
    get the fitness of the evolved frz map
    rank the frz maps based on the fitness
    discard the worst frz maps and return the best frz maps
    '''
    rng, _rng = jax.random.split(rng)
    frz_rng = jax.random.split(_rng, config.evo_pop_size)
    
    frz_maps = frz_maps[:config.evo_pop_size]
    mutate_fn = jax.vmap(mutate_frz_map, in_axes=(0, 0, None))
    mutant_frz_maps = mutate_fn(frz_rng, frz_maps, config)
    frz_maps = jnp.concatenate((frz_maps, mutant_frz_maps), axis=0)
 
    def eval_frzs(frz_maps, network_params):
        frz_maps = jnp.tile(
            frz_maps, (int(np.ceil(config.n_envs / frz_maps.shape[0])), 1, 1)
        )[:config.n_envs]
        queued_state = QueuedState(ctrl_trgs=jnp.zeros(len(env.prob.stat_trgs)))
        queued_state = jax.vmap(env.queue_frz_map, in_axes=(None, 0))(queued_state, frz_maps)
        eval_rng = jax.random.split(rng, config.n_envs)

        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None, 0))(
                eval_rng, env_params, queued_state
        )

        _, (states, rewards, dones, infos, values) = jax.lax.scan(
            step_env_evo_eval, (rng, obsv, env_state, network_params),
            None, 1*env.max_steps)
        
        # regret value
        def calc_regret_value(carry, t_step):
            '''
            for each env (axis = 0)
            rewards = [r1, r2, r3, r4, ..., ]
            discount_factors = [gamma^0, ^1, ^2, ^3, ..., ]
            values = [v1, v2, v3, v4, ..., ]
            '''
            rewards, discount_factors, values = carry
            breakpoint()
            discount_factors = discount_factors[::-1][:t_step]
            rewards, values = rewards[:t_step, ...], values[:t_step, ...]
            return jnp.abs(rewards * discount_factors - values) 

        discount_factors = jnp.power(config.GAMMA, jnp.arange(values.shape[0]))

        _, fits = jax.lax.scan(calc_regret_value, (rewards, discount_factors, values), jnp.arange(values.shape[0]))
        # fits = jax.lax.fori_loop(0, values.shape[0], calc_regret_value, (rewards, discount_factors, values))
        fits = fits.sum(axis=0)
        return fits, states

    def step_env_evo_eval(carry, _):
        rng_r, obs_r, env_state_r, network_params = carry
        rng_r, _rng_r = jax.random.split(rng_r)

        pi, value = network.apply(network_params, obs_r)
        action_r = pi.sample(seed=rng_r)
        action_r = jnp.full(action_r.shape, 0) # FIXME dumdum Debugging evo 

        rng_step = jax.random.split(_rng_r, config.n_envs)

        # rng_step_r = rng_step_r.reshape((config.n_gpus, -1) + rng_step_r.shape[1:])
        vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        # pmap_step_fn = jax.pmap(vmap_step_fn, in_axes=(0, 0, 0, None))
        obs_r, env_state_r, reward_r, done_r, info_r = vmap_step_fn(
                        rng_step, env_state_r, action_r,
                        env_params)
        
        return (rng_r, obs_r, env_state_r, network_params),\
            (env_state_r, reward_r, done_r, info_r, value)
    
    fits, states = eval_frzs(frz_maps, network_params)    
    fits = fits.reshape((-1, config.evo_pop_size*2)).mean(axis=0)
    # sort the top frz maps based on the fitness
    # Get indices of the top 5 largest elements
    top_indices = jnp.argpartition(-fits, config.evo_pop_size)[:config.evo_pop_size] # We negate arr to get largest elements
    top = frz_maps[:2 * config.evo_pop_size][top_indices]
    
    top_fitnesses = fits[top_indices]
    # evo_writer = SummaryWriter(os.path.join(get_exp_dir(config), "evo"))
    # jax.debug.breakpoint()
    # evo_writer.add_scalar("fitness", top_fitnesses.mean(0), runner_state.update_i)
    return EvoState(top_fitness=top_fitnesses, frz_map=top) # here do I need to init an empty one and evo_state.replace(top_fitness=top_fitnesses, frz_map=top) ?
    # return top
    

        

    
def mutate_frz_map(rng, frz_map, config: TrainConfig):
    '''
    mutate the frz maps
    '''
    mut_tiles = jax.random.bernoulli(
        rng, p=config.evo_mutate_prob, shape=frz_map.shape)
    # frz_map = (frz_map + mut_tiles) % 2
    new_frz_map = jnp.logical_xor(frz_map, mut_tiles)
    return new_frz_map
    
