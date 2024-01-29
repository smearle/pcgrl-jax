import json
import os

import chex
from flax import struct
import hydra
import imageio
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from config import EvalConfig
from envs.pcgrl_env import gen_dummy_queued_state
from envs.probs.problem import ProblemState
from purejaxrl.experimental.s5.wrappers import LogWrapper
from train import init_checkpointer
from utils import get_exp_dir, get_network, gymnax_pcgrl_make, init_config


@struct.dataclass
class EvalData:
    # cell_losses: chex.Array
    # cell_progs: chex.Array
    # cell_rewards: chex.Array
    mean_ep_reward: chex.Array

@hydra.main(version_base=None, config_path='./', config_name='eval_pcgrl')
def main_eval(config: EvalConfig):
    config = init_config(config)

    exp_dir = get_exp_dir(config)
    if not config.random_agent:
        checkpoint_manager, restored_ckpt = init_checkpointer(config)
        network_params = restored_ckpt['runner_state'].train_state.params
    elif not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env = LogWrapper(env)
    env.prob.init_graphics()
    network = get_network(env, env_params, config)

    rng = jax.random.PRNGKey(42)
    reset_rng = jax.random.split(rng, config.n_eval_envs)

    def eval(env_params):
        # obs, env_state = env.reset(reset_rng, env_params)
        queued_state = gen_dummy_queued_state(env)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None, None))(
            reset_rng, env_params, queued_state)

        def step_env(carry, _):
            rng, obs, env_state = carry
            rng, rng_act = jax.random.split(rng)
            if config.random_agent:
                action = env.action_space(env_params).sample(rng_act)
            else:
                # obs = jax.tree_map(lambda x: x[None, ...], obs)
                action = network.apply(network_params, obs)[0].sample(seed=rng_act)

            rng_step = jax.random.split(rng, config.n_eval_envs)
            obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                rng_step, env_state, action, env_params
            )
            # frame = env.render(env_state)
            # Can't concretize these values inside jitted function (?)
            # So we add the stats on cpu later (below)
            # frame = render_stats(env, env_state, frame)
            return (rng, obs, env_state), (env_state, reward, done, info)

        print('Scanning episode steps:')
        _, (states, rewards, dones, infos) = jax.lax.scan(
            step_env, (rng, obs, env_state), None,
            length=config.n_eps*env.max_steps)

        return _, (states, rewards, dones, infos)

    def _eval(env_params):
        _, (states, reward, dones, infos) = eval(env_params)

        return states, reward, dones

    json_path = os.path.join(exp_dir, 'stats.json')

    # For each bin, evaluate the change pct. at the center of the bin
    
    if config.reevaluate:
        states, rewards, dones = _eval(env_params)

        # Everything has size (n_bins, n_steps, n_envs)
        # Mask out so we only have the final step of each episode
        ep_rews = states.returned_episode_returns * dones
        # Get mean episode reward for each bin
        ep_rews = jnp.sum(ep_rews)
        mean_ep_rew = ep_rews / jnp.sum(dones)

        # cell_loss = jnp.mean(jnp.where(dones, 
        #                         jnp.sum(
        #                             jnp.abs(states.ctrl_trgs - states.stats) * env.prob.stat_weights
        #                             , axis=-1)
        #                         , 0))

        # Compute weighted loss from targets
        # cell_loss = jnp.mean(jnp.abs(
        #     final_prob_states.ctrl_trgs - final_prob_states.stats) * env.prob.stat_weights
        # )

        # Compute relative progress toward target from initial metric values
        # cell_progs = 1 - jnp.abs(final_stats[:, ctrl_metrics] - ctrl_trg) / jnp.abs(init_stats[:, ctrl_metrics] - ctrl_trg)
        # final_prog = jnp.where(done_idxsjnp.abs(cell_states.stats - final_prob_states.ctrl_trgs)
        # trg_prog = jnp.abs(init_prob_states.stats - init_prob_states.ctrl_trgs)
        # trg_prog = jnp.where(trg_prog == 0, 1e4, trg_prog)
        # cell_progs = (1 - jnp.abs(final_prog / trg_prog))
        # cell_prog = jnp.mean(cell_progs)

        stats = EvalData(
            mean_ep_reward=mean_ep_rew,
        )

        with open(json_path, 'w') as f:
            json_stats = {k: v.tolist() for k, v in stats.__dict__.items()}
            json.dump(json_stats, f, indent=4)
    else:
        with open(json_path, 'r') as f:
            stats = json.load(f)
            stats = EvalData(**stats)
    
if __name__ == '__main__':
    main_eval()