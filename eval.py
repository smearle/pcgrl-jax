import copy
import json
import os
from typing import Optional

import chex
from flax import struct
import hydra
import imageio
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from tensorflow_probability.python.internal.backend.jax.numpy_logging import warning

from conf.config import EvalConfig, TrainConfig
from envs.pcgrl_env import gen_dummy_queued_state
from envs.probs.problem import ProblemState, get_loss
from purejaxrl.experimental.s5.wrappers import LogWrapper, LossLogWrapper
from train import init_checkpointer
from utils import get_exp_dir, init_network, gymnax_pcgrl_make, init_config


@struct.dataclass
class EvalData:
    # cell_losses: chex.Array
    # cell_progs: chex.Array
    # cell_rewards: chex.Array
    mean_ep_reward: chex.Array
    mean_min_ep_loss: chex.Array
    min_min_ep_loss: chex.Array
    n_eval_eps: int
    n_parameters: Optional[int] = None

@hydra.main(version_base=None, config_path='./conf', config_name='eval_pcgrl')
def main_eval(eval_config: EvalConfig = None):
    # if not hasattr(eval_config, 'INIT_CONFIG'):
    #    eval_config = init_config(eval_config)

    exp_dir = eval_config.exp_dir
    if not eval_config.random_agent:
        print(f'Attempting to load checkpoint from {exp_dir}')
        checkpoint_manager, restored_ckpt = init_checkpointer(eval_config)
        network_params = restored_ckpt['runner_state'].train_state.params
    elif not os.path.exists(exp_dir):
        warning(f'No checkpoint found at {exp_dir}. Initializing network from scratch.')
        network_params = network.init(rng, init_x)
        os.makedirs(exp_dir)
    else:
        pass
        # network_params = network.init(rng, init_x)
        # print(network_params)

    # print(network_params)

    # Preserve the config as it was during training (minus `eval_` hyperparams), for future reference
    train_config = copy.deepcopy(eval_config)
    eval_config = init_config_for_eval(eval_config)
    env, env_params = gymnax_pcgrl_make(eval_config.env_name, config=eval_config)
    env = LossLogWrapper(env)
    env.prob.init_graphics()
    network = init_network(env, env_params, eval_config)
    rng = jax.random.PRNGKey(eval_config.eval_seed)

    init_x = env.gen_dummy_obs(env_params)
    # init_x = env.observation_space(env_params).sample(_rng)[None]
    # model_summary = network.subnet.tabulate(rng, init_x.map_obs, init_x.flat_obs)
    n_parameters = sum(np.prod(p.shape) for p in jax.tree_leaves(network_params) if isinstance(p, jnp.ndarray))

    reset_rng = jax.random.split(rng, eval_config.n_eval_envs)

    def eval(env_params):
        queued_state = gen_dummy_queued_state(env)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None, None))(
            reset_rng, env_params, queued_state)

        def step_env(carry, _):
            rng, obs, env_state = carry
            rng, rng_act = jax.random.split(rng)
            if eval_config.random_agent:
                action = env.action_space(env_params).sample(rng_act)
            else:
                action = network.apply(network_params, obs)[0].sample(seed=rng_act)

            rng_step = jax.random.split(rng, eval_config.n_eval_envs)
            obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                rng_step, env_state, action, env_params
            )
            return (rng, obs, env_state), (env_state, reward, done, info)

        print('Scanning episode steps:')
        _, (states, rewards, dones, infos) = jax.lax.scan(
            step_env, (rng, obs, env_state), None,
            length=eval_config.n_eps*env.max_steps)

        return _, (states, rewards, dones, infos)

    def _eval(env_params):
        _, (states, reward, dones, infos) = eval(env_params)

        return states, reward, dones

    stats_name = \
        "stats" + \
        get_eval_name(eval_config=eval_config, train_config=train_config) + \
        f".json"
    json_path = os.path.join(exp_dir, stats_name)

    # For each bin, evaluate the change pct. at the center of the bin

    if eval_config.reevaluate or not os.path.exists(json_path):
        states, rewards, dones = _eval(env_params)

        stats = get_eval_stats(states, dones)
        stats = stats.replace(
            n_parameters=n_parameters,
        )

        with open(json_path, 'w') as f:
            json_stats = {k: v.tolist() for k, v in stats.__dict__.items() if isinstance(v, jnp.ndarray)}
            json.dump(json_stats, f, indent=4)
    else:
        with open(json_path, 'r') as f:
            stats = json.load(f)
            stats = EvalData(**stats)

    jax.block_until_ready(stats)

def get_eval_stats(states, dones):
    # Everything has size (n_bins, n_steps, n_envs)
    # Mask out so we only have the final step of each episode
    ep_rews = states.log_env_state.returned_episode_returns * dones
    # Get mean episode reward
    ep_rews = jnp.sum(ep_rews)
    n_eval_eps = jnp.sum(dones)
    mean_ep_rew = ep_rews / n_eval_eps

    # Get the average min. episode loss
    min_ep_losses = states.min_episode_losses
    # Mask out so we only have the final step of each episode
    min_ep_losses = jnp.where(dones, min_ep_losses, jnp.nan)
    # Get mean episode loss
    sum_min_ep_losses = jnp.nansum(min_ep_losses)
    mean_min_ep_loss = sum_min_ep_losses / n_eval_eps
    min_min_ep_loss = jnp.nanmin(min_ep_losses)

    stats = EvalData(
        mean_ep_reward=mean_ep_rew,
        mean_min_ep_loss=mean_min_ep_loss,
        min_min_ep_loss=min_min_ep_loss,
        n_eval_eps=n_eval_eps,
    )
    return stats


def init_config_for_eval(config):
    if config.eval_map_width is not None:
        config.map_width = config.eval_map_width
    if config.eval_max_board_scans is not None:
        config.max_board_scans = config.eval_max_board_scans
    if config.eval_randomize_map_shape is not None:
        config.randomize_map_shape = config.eval_randomize_map_shape
    return config


def get_eval_name(eval_config: EvalConfig, train_config: TrainConfig):
    """Get a name for the eval stats file, based on the eval hyperparams.

    If eval_config has been initialized for eval (with standard hyperparameters being replaced by their `eval_`
    counterparts), then we will check against train_config, in case we have saved stats using the same hyperparameters,
    but without having explicitly specified them as eval hyperparameters. Otherwise eval_config and train_config can be
    the same config (with differing train/eval hyperparams).
    """
    eval_name = \
        (f"_randMap-{eval_config.eval_randomize_map_shape}" if
         eval_config.eval_randomize_map_shape is not None and
         # This is for backward compatibility, in terms of being able to re-use prior eval stats jsons.
         eval_config.eval_randomize_map_shape != train_config.randomize_map_shape
         else "") + \
        (f"_w-{eval_config.eval_map_width}" if eval_config.eval_map_width is not None else "") + \
        (f"_bs-{eval_config.eval_max_board_scans}" if eval_config.eval_max_board_scans is not None else "") + \
        (f"_seed-{eval_config.eval_seed}" if eval_config.eval_seed is not None else "")
    return eval_name


if __name__ == '__main__':
    main_eval()