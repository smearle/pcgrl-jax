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

from conf.config import EvalConfig
from envs.pcgrl_env import render_stats
from train import init_checkpointer
from utils import get_exp_dir, init_network, gymnax_pcgrl_make, init_config


@struct.dataclass
class EvalData:
    ctrl_trgs: chex.Array
    cell_losses: chex.Array
    cell_progs: chex.Array
    cell_rewards: chex.Array

@hydra.main(version_base=None, config_path='./', config_name='eval_pcgrl')
def main_eval_ctrls(config: EvalConfig):
    config = init_config(config)

    exp_dir = config.exp_dir
    if not config.random_agent:
        checkpoint_manager, restored_ckpt = init_checkpointer(config)
        network_params = restored_ckpt['runner_state'].train_state.params
    elif not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env.prob.init_graphics()
    network = init_network(env, env_params, config)

    rng = jax.random.PRNGKey(42)
    reset_rng = jax.random.split(rng, config.n_envs)

    ctrl_metrics = env.prob.ctrl_metrics

    def eval_ctrls(ctrl_trgs):
        # obs, env_state = env.reset(reset_rng, env_params)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            reset_rng, env_params)
        env_state = env_state.replace(prob_state=env_state.prob_state.replace(
            ctrl_trgs=env_state.prob_state.ctrl_trgs.at[
                :, env.prob.ctrl_metrics].set(ctrl_trgs)))

        def step_env(carry, _):
            rng, obs, env_state = carry
            rng, rng_act = jax.random.split(rng)
            if config.random_agent:
                action = env.action_space(env_params).sample(rng_act)
            else:
                # obs = jax.tree_map(lambda x: x[None, ...], obs)
                action = network.apply(network_params, obs)[0].sample(seed=rng_act)

            rng_step = jax.random.split(rng, config.n_envs)
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

    # Bin up each ctrl metric into 10 bins
    # For each bin, evaluate the ctrl metric at the center of the bin
    metric_bounds = env.prob.metric_bounds[env.prob.ctrl_metrics]
    ctrl_trgs = np.linspace(
        metric_bounds[:, 0],
        metric_bounds[:, 1],
        config.n_bins,
    )
    if len(env.prob.ctrl_metrics) == 2:
        # take cartesian product of ctrl_trgs
        ctrl_trg_pairs = np.array(np.meshgrid(*ctrl_trgs.T)).T.reshape(-1, len(env.prob.ctrl_metrics)) 
    elif len(env.prob.ctrl_metrics) == 1:
        ctrl_trg_pairs = ctrl_trgs[:, 0]
    else:
        raise NotImplementedError

    im_shape = tuple([config.n_bins] * len(env.prob.ctrl_metrics))
    if len(im_shape) == 1:
        im_shape += (1,)

    # for i, ctrl_trgs in enumerate(ctrl_trg_pairs):
    def _eval_ctrls(ctrl_trg):
        _, (states, reward, dones, infos) = eval_ctrls(ctrl_trg)
        ep_rewards = reward.sum(axis=0)
        cell_reward = jnp.mean(ep_rewards)

        cell_stats = states.prob_state.stats
        init_stats = cell_stats[0]
        final_stats = cell_stats[-1]
        cell_loss = jnp.mean(jnp.abs(final_stats[:, ctrl_metrics] - ctrl_trg))

        # Compute relative progress toward target from initial metric values
        cell_progs = 1 - jnp.abs(final_stats[:, ctrl_metrics] - ctrl_trg) / jnp.abs(init_stats[:, ctrl_metrics] - ctrl_trg)
        cell_prog = jnp.mean(cell_progs)

        eval_data = EvalData(
            ctrl_trgs=ctrl_trg,
            cell_losses=cell_loss,
            cell_rewards=cell_reward,
            cell_progs = cell_prog,
        )
        
        return eval_data

    json_path = os.path.join(exp_dir, 'ctrl_stats.json')
    
    if config.reevaluate:
        stats = jax.vmap(_eval_ctrls)(ctrl_trg_pairs)

        with open(json_path, 'w') as f:
            json_stats = {k: v.tolist() for k, v in stats.__dict__.items()}
            json.dump(json_stats, f, indent=4)
    else:
        with open(json_path, 'r') as f:
            stats = json.load(f)
            stats = EvalData(**stats)

    cell_progs = np.array(stats.cell_progs)
    cell_progs = cell_progs.reshape(im_shape)

    fig, ax = plt.subplots()
    ax.imshow(cell_progs)
    if len(im_shape) == 1:
        ax.set_xticks([])
    elif len(im_shape) == 2:
        ax.set_xticks(np.arange(len(ctrl_trgs[0])))
        ax.set_xticklabels(ctrl_trgs[0])
    ax.set_yticks(np.arange(len(ctrl_trgs))) 
    ax.set_yticklabels(ctrl_trgs[:, 0])
    fig.colorbar()
    # plt.imshow(cell_progs)
    # plt.colorbar()
    # plt.title('Control target success')

    plt.savefig(os.path.join(exp_dir, 'ctrl_loss.png'))

if __name__ == '__main__':
    main_eval_ctrls()
