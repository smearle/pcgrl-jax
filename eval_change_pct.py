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
from envs.pcgrl_env import gen_dummy_queued_state
from envs.probs.problem import ProblemState
from purejaxrl.experimental.s5.wrappers import LogWrapper
from train import init_checkpointer
from utils import get_exp_dir, init_network, gymnax_pcgrl_make, init_config


# When we evaluate without enforcing change percentage, we allow this many board scans
EVAL_MAX_BOARD_SCANS = 5


def get_change_pcts(n_bins):
    # For each bin, evaluate the change pct. at the center of the bin
    change_pcts = np.linspace(0.1, 1, n_bins)
    change_pcts = np.concatenate([change_pcts, [-1]])
    return change_pcts


@struct.dataclass
class EvalData:
    # cell_losses: chex.Array
    # cell_progs: chex.Array
    # cell_rewards: chex.Array
    cell_rewards: chex.Array

@hydra.main(version_base="1.3", config_path='./', config_name='eval_pcgrl')
def main_eval_cp(config: EvalConfig):
    config = init_config(config)

    exp_dir = config.exp_dir
    if not config.random_agent:
        checkpoint_manager, restored_ckpt = init_checkpointer(config)
        network_params = restored_ckpt['runner_state'].train_state.params
    elif not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Setting env params for eval. Note that this must come after loading the
    # checkpoint lest we try to load a non-existent experiment.
    config.max_board_scans = EVAL_MAX_BOARD_SCANS

    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env = LogWrapper(env)
    env.prob.init_graphics()
    network = init_network(env, env_params, config)

    rng = jax.random.PRNGKey(42)
    reset_rng = jax.random.split(rng, config.n_eval_envs)

    def eval_cp(change_pct, env_params):
        # obs, env_state = env.reset(reset_rng, env_params)
        queued_state = gen_dummy_queued_state(env)
        env_params = env_params.replace(change_pct=change_pct)
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

    def _eval_cp(change_pct, env_params):
        _, (states, reward, dones, infos) = eval_cp(change_pct, env_params)

        return states, reward, dones


    json_path = os.path.join(exp_dir, 'cp_stats.json')

    change_pcts = get_change_pcts(config.n_bins)
    
    if config.reevaluate:
        states, reward, dones = jax.vmap(_eval_cp, in_axes=(0, None))(change_pcts, env_params)

        # Everything has size (n_bins, n_steps, n_envs)
        # Mask out so we only have the final step of each episode
        ep_rews = states.returned_episode_returns * dones
        # Get mean episode reward for each bin
        ep_rews = jnp.sum(ep_rews, axis=(1, 2))
        mean_ep_rews = ep_rews / jnp.sum(dones, axis=(1, 2))

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
            cell_rewards=mean_ep_rews,
        )

        with open(json_path, 'w') as f:
            json_stats = {k: v.tolist() for k, v in stats.__dict__.items()}
            json.dump(json_stats, f, indent=4)
    else:
        with open(json_path, 'r') as f:
            stats = json.load(f)
            stats = EvalData(**stats)
    
    # Make a bar plot of cell losses
    # fig, ax = plt.subplots()
    # ax.bar(np.arange(len(stats.cell_losses)), stats.cell_losses)
    # ax.set_xticks(np.arange(len(stats.cell_losses)))
    # ax.set_xticklabels([f'{cp:.2f}' for cp in change_pcts])
    # ax.set_ylabel('Loss')
    # ax.set_xlabel('Change pct.')
    # plt.savefig(os.path.join(exp_dir, 'cp_loss.png'))

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(stats.cell_rewards)), stats.cell_rewards)
    ax.set_xticks(np.arange(len(stats.cell_rewards)))
    ax.set_xticklabels([f'{cp:.2f}' for cp in change_pcts])
    ax.set_ylabel('Reward')
    ax.set_xlabel('Change pct.')
    plt.savefig(os.path.join(exp_dir, 'cp_reward.png'))

    # cell_progs = np.array(stats.cell_progs)

    # fig, ax = plt.subplots()
    # ax.imshow(cell_progs)
    # if len(im_shape) == 1:
    #     ax.set_xticks([])
    # elif len(im_shape) == 2:
    #     ax.set_xticks(np.arange(len(ctrl_trgs[0])))
    #     ax.set_xticklabels(ctrl_trgs[0])
    # ax.set_yticks(np.arange(len(ctrl_trgs))) 
    # ax.set_yticklabels(ctrl_trgs[:, 0])
    # fig.colorbar()
    # plt.imshow(cell_progs)
    # plt.colorbar()
    # plt.title('Control target success')


if __name__ == '__main__':
    main_eval_cp()
