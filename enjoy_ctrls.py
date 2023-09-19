import os

import hydra
import imageio
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from config import EvalConfig
from envs.pcgrl_env import render_stats
from train import init_checkpointer
from utils import get_exp_dir, get_network, gymnax_pcgrl_make, init_config


@hydra.main(version_base=None, config_path='./', config_name='enjoy_pcgrl')
def main_enjoy_ctrls(config: EvalConfig):
    config = init_config(config)

    exp_dir = get_exp_dir(config)
    if not config.random_agent:
        checkpoint_manager, restored_ckpt = init_checkpointer(config)
        network_params = restored_ckpt['runner_state'].train_state.params
    elif not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env.prob.init_graphics()
    network = get_network(env, env_params, config)

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
            frame = env.render(env_state)
            # Can't concretize these values inside jitted function (?)
            # So we add the stats on cpu later (below)
            frame = render_stats(env, env_state, frame)
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
    def _eval_ctrls(ctrl_trgs):
        _, (states, reward, dones, infos) = eval_ctrls(ctrl_trgs)

        cell_stats = states.prob_state.stats
        cell_loss = jnp.sum(jnp.abs(cell_stats[ctrl_metrics] - ctrl_trgs))
        
        # print(f'Cell loss: {cell_loss}')
        return cell_loss
    
    cell_losses = jax.vmap(_eval_ctrls)(ctrl_trg_pairs)

    cell_losses = cell_losses.reshape(im_shape)
    plt.imshow(cell_losses)
    plt.colorbar()
    plt.title('Control target success')

    
    plt.savefig(os.path.join(exp_dir, 'ctrl_loss.png'))

if __name__ == '__main__':
    main_enjoy_ctrls()
