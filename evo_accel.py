'''
get the fitness of the evolved frz map (or other thingys we want to evolve)
'''
import jax
from jax import numpy as jnp
import numpy as np
from config import TrainConfig


def apply_evo(rng, frz_maps, env, env_params, network_params, network, config: TrainConfig):
    '''
    copy and mutate the frz maps
    get the fitness of the evolved frz map
    rank the frz maps based on the fitness
    discard the worst frz maps and return the best frz maps
    '''
    rng, _rng = jax.random.split(rng)
    frz_rng = jax.random.split(_rng, config.evo_pop_size)
    reset_rng = jax.random.split(rng, config.n_envs)
    
    # frz_maps = jnp.repeat(frz_maps, 2, axis=0)
    mutate_fn = jax.vmap(mutate_frz_map, in_axes=(0, 0))
    mutant_frz_maps = mutate_fn(frz_rng, frz_maps[:config.evo_pop_size//2])
    frz_maps = jnp.concatenate((frz_maps, mutant_frz_maps), axis=0)

    set_frz_map_fn = jax.vmap(env.set_frz_map, in_axes=(0,))(frz_maps)

    vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None))
    obsv, env_state = vmap_reset_fn(reset_rng, env_params)
    
 
    def eval_frzs(frz_maps, network_params):
        _, (states, rewards, dones, infos, fits) = jax.lax.scan(
            step_env_evo_eval, (rng, obsv, env_state, network_params),
            None, 1*env.max_steps)

        return fits, states

    def step_env_evo_eval(carry, _):
        rng_r, obs_r, env_state_r, network_params = carry
        rng_r, _rng_r = jax.random.split(rng_r)

        pi, value = network.apply(network_params, obs_r)
        action_r = pi.sample(seed=rng_r)

        rng_step = jax.random.split(_rng_r, config.n_render_eps)

        # rng_step_r = rng_step_r.reshape((config.n_gpus, -1) + rng_step_r.shape[1:])
        vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        # pmap_step_fn = jax.pmap(vmap_step_fn, in_axes=(0, 0, 0, None))
        obs_r, env_state_r, reward_r, done_r, info_r = vmap_step_fn(
                        rng_step, env_state_r, action_r,
                        env_params)
        return (rng_r, obs_r, env_state_r, network_params),\
            (env_state_r, reward_r, done_r, info_r)
    
    fits, states = eval_frzs(frz_maps, network_params)    
    # sort the top frz maps based on the fitness
    # Get indices of the top 5 largest elements
    top_indices = np.argpartition(-fits, config.evo_pop_size)[:config.evo_pop_size] # We negate arr to get largest elements
    top = frz_maps[top_indices]

    return top
    

        

    
def mutate_frz_map(rng, frz_map, config: TrainConfig):
    '''
    mutate the frz maps
    '''
    mut_tiles = jax.random.bernoulli(
        rng, p=config.evo_mutate_prob, shape=frz_map.shape)
    frz_map = (frz_map + mut_tiles) % 2
    return frz_map
    
