from functools import partial
import math
import os
from timeit import default_timer
import PIL

import chex
from flax import struct
import hydra
import jax
import jax.numpy as jnp
import numpy as np

from config import EvoMapConfig
from envs.pcgrl_env import PCGRLEnv, PCGRLEnvState
from envs.probs.problem import ProblemState, get_loss
from utils import gymnax_pcgrl_make, init_config_evo_map


@struct.dataclass
class EvoMapState:
    maps: chex.Array
    map_losses: chex.Array
    map_prob_states: ProblemState
    gen_i: int
    rng: jax.random.PRNGKey


@hydra.main(version_base=None, config_path="./", config_name="evo_map_pcgrl")
def evolve_main(config: EvoMapConfig):
    config = init_config_evo_map(config)
    os.makedirs(os.path.join(config.exp_dir, 'renders'), exist_ok=True)
    os.makedirs(os.path.join(config.exp_dir, 'ckpts'), exist_ok=True)
    rng = jax.random.PRNGKey(config.seed)

    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env.init_graphics()
    _, env_state = env.reset(rng, env_params)

    evolve_jit = jax.jit(make_evolve(config))

    start_time = default_timer()
    evo_state: EvoMapState = evolve_jit(rng)
    print(f"Evolution took {default_timer() - start_time} seconds.")
    log(evo_state)
    render(evo_state, env_state, env, config)


def make_evolve(config: EvoMapConfig):
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env.init_graphics()
    _render = partial(render, env=env, config=config)

    def evolve(rng: jax.random.PRNGKey):
        # FIXME: Need this for rendering currently. But shouldn't have to reset like this. Construct state manually, or (better) make rendering not depend directly 
        # on state.
        _, env_state = env.reset(rng, env_params)

        # init_maps, _ = env.prob.gen_init_map(rng)
        init_rng = jax.random.split(rng, config.evo_pop_size)
        maps = jax.vmap(env.prob.gen_init_map, in_axes=(0,))(init_rng).env_map
        # Evaluate the maps
        map_stats = jax.vmap(env.prob.get_curr_stats, in_axes=(0,))(maps)
        # Get the loss for each map
        map_losses = jax.vmap(get_loss, in_axes=(0, None, None, None, None))(
            map_stats.stats, env.prob.stat_weights, env.prob.stat_trgs, env.prob.ctrl_threshes, env.prob.metric_bounds)

        # Sort the maps by loss
        top_idxs = jnp.argsort(map_losses)
        maps = maps[top_idxs]
        map_losses = map_losses[top_idxs]
        map_stats = jax.tree_map(lambda x: x[top_idxs], map_stats)

        # Prepare state for scan loop
        initial_state = EvoMapState(maps, map_losses, map_stats, 0, rng)


        def evolve_step(state: EvoMapState, _):
            maps, map_losses, map_prob_states, gen_i, rng = state.maps, state.map_losses, state.map_prob_states, state.gen_i, state.rng

            rng, _ = jax.random.split(rng)

            # Select parents
            parents = maps[:config.n_parents]
            # Tile the parents to match the population size
            parents = jnp.tile(parents, (math.ceil(config.evo_pop_size / config.n_parents), 1, 1))[:config.evo_pop_size]

            # Mutate the top maps
            mut_rng = jax.random.split(rng, config.evo_pop_size)
            children = jax.vmap(mutate_map, in_axes=(0, 0, None, None, None))(mut_rng, parents, config.mut_rate, env.prob.tile_enum, env.prob.tile_probs)

            # Evaluate the maps
            children_states = jax.vmap(env.prob.get_curr_stats, in_axes=(0,))(children)
            # Get the loss for each map
            mut_map_losses = jax.vmap(get_loss, in_axes=(0, None, None, None, None))(
                children_states.stats, env.prob.stat_weights, env.prob.stat_trgs, env.prob.ctrl_threshes, env.prob.metric_bounds)

            maps = jnp.concatenate([maps, children])
            map_losses = jnp.concatenate([map_losses, mut_map_losses])
            map_prob_states = jax.tree_map(
                lambda x, y: jnp.concatenate([x, y]),
                map_prob_states,
                children_states)

            # Get the top evo_pop_size maps
            top_idxs = jnp.argpartition(map_losses, config.evo_pop_size)[:config.evo_pop_size]
            maps = maps[top_idxs]
            map_losses = map_losses[top_idxs]
            map_prob_states = jax.tree_map(lambda x: x[top_idxs], map_prob_states)

            state = EvoMapState(maps, map_losses, map_prob_states, gen_i + 1, rng)

            # Log and render conditionally
            if config.callbacks:
                should_log = (gen_i % config.log_freq == 0)
                jax.lax.cond(
                    should_log, 
                    partial(jax.debug.callback, log), 
                    lambda _: None, 
                    state)
                should_render = (gen_i % config.render_freq == 0)
                jax.lax.cond(
                    should_render, 
                    partial(jax.debug.callback, _render), 
                    lambda _, __: None, 
                    state, env_state)

            return state, None  # No output needed per iteration

        final_state, _ = jax.lax.scan(evolve_step, initial_state, jnp.arange(config.n_generations))
        return final_state

        # Final logging and possibly rendering outside the scan loop if needed

    return lambda rng: evolve(rng)


@partial(jax.jit, static_argnames=("tile_enum",))
def mutate_map(rng, map, mut_rate, tile_enum, tile_probs):
    mut_rng, _ = jax.random.split(rng)
    mut_rate = jax.random.uniform(mut_rng, minval=0.0, maxval=mut_rate, shape=(1,))
    mut_rng, tile_rng = jax.random.split(rng)
    mut_map = jax.random.bernoulli(mut_rng, mut_rate, map.shape)
    mut_map = jnp.where(mut_map, jax.random.choice(tile_rng, tile_probs, map.shape), map)
    mut_map = mut_map.astype(jnp.int32)
    return mut_map


def log(evo_state: EvoMapState):
    map_losses = evo_state.map_losses
    i = evo_state.gen_i - 1
    print(f"Generation {i}: Best loss: {jnp.min(map_losses)}. Mean loss: {jnp.mean(map_losses)}")


def render(evo_state: EvoMapState, env_state: PCGRLEnvState, env: PCGRLEnv,
           config: EvoMapConfig):
    maps, map_prob_states = evo_state.maps, evo_state.map_prob_states
    i = evo_state.gen_i - 1

    # Manually batching our single env state to make it compatible with render FIXME
    env_states = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None], maps.shape[0], axis=0), env_state)
    env_states = env_states.replace(env_map=maps)
    env_states = env_states.replace(
        prob_state=map_prob_states
    )
    frames = jax.vmap(env.render, in_axes=(0))(env_states)

    os.makedirs(os.path.join(config.exp_dir, "renders", f"gen_{i}"),
                exist_ok=True
    )
    for j, frame in enumerate(frames):
        frame = PIL.Image.fromarray(np.array(frame))
        frame.save(os.path.join(
            config.exp_dir,
            "renders",
            f"gen_{i}",
            f"map_{j}.png"))




if __name__ == '__main__':
    evolve_main()