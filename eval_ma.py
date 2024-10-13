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
import orbax.checkpoint as ocp

from conf.config import TrainConfig, MultiAgentEvalConfig
from envs.pcgrl_env import PCGRLEnv, gen_dummy_queued_state
from envs.probs.problem import ProblemState, get_loss
from ma_utils import batchify, ma_init_config, init_run, restore_run, unbatchify
from marl.model import ScannedRNN
from marl.wrappers.baselines import MALossLogWrapper, MultiAgentWrapper
from purejaxrl.experimental.s5.wrappers import LogWrapper, LossLogWrapper
from train import init_checkpointer
from utils import get_env_params_from_config


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
def main_eval_ma(eval_config: MultiAgentEvalConfig):
    ma_init_config(eval_config)
    rng = jax.random.PRNGKey(eval_config.eval_seed)

    exp_dir = eval_config.exp_dir

    options = ocp.CheckpointManagerOptions(
        max_to_keep=2, create=True)
    ckpt_manager = ocp.CheckpointManager(
        eval_config._ckpt_dir, 
        # ocp.PyTreeCheckpointer(), 
        options=options)
    latest_update_step = ckpt_manager.latest_step()

    # if not eval_config.random_agent:
    print(f'Attempting to load checkpoint from {exp_dir}')
    runner_state, actor_network, env, latest_update_step = init_run(
        eval_config, ckpt_manager, latest_update_step, rng)
    runner_state, wandb_run_id = restore_run(
        eval_config, runner_state, ckpt_manager, latest_update_step
    )
    # elif not os.path.exists(exp_dir):
    #     network_params = network.init(rng, init_x)
    #     os.makedirs(exp_dir)

    # Preserve the config as it was during training (minus `eval_` hyperparams), for future reference
    train_config = copy.deepcopy(eval_config)
    eval_config = init_config_for_eval(eval_config)
    # env, env_params = gymnax_pcgrl_make(eval_config.env_name, config=eval_config)

    env_params = get_env_params_from_config(eval_config)
    env = PCGRLEnv(env_params)

    # Wrap environment with JAXMARL wrapper
    env = MultiAgentWrapper(env, env_params)

    # Wrap environment with LogWrapper
    env = MALossLogWrapper(env)

    # env.prob.init_graphics()
    # network = init_network(env, env_params, eval_config)
    rng, _rng_actor = jax.random.split(rng, 2)

    ac_init_hstate = ScannedRNN.initialize_carry(eval_config._num_eval_actors, eval_config.hidden_dims[0])

    hstate = ac_init_hstate
    actor_network_params = runner_state.train_states[0].params
    n_parameters = sum(np.prod(p.shape) for p in jax.tree_leaves(actor_network_params) if isinstance(p, jnp.ndarray))

    reset_rng = jax.random.split(rng, eval_config.n_eval_envs)

    def eval(env_params):
        queued_state = gen_dummy_queued_state(env)
        obs, env_state = jax.vmap(env.reset, in_axes=(0))(
            reset_rng)
        # done = {agent: jnp.zeros((eval_config.n_eval_envs), dtype=bool) for agent in env.agents + ['__all__']}
        done = jnp.zeros((eval_config._num_eval_actors), dtype=bool)

        def step_env(carry, _):
            rng, obs, last_done, env_state, hstate = carry
            rng, rng_act = jax.random.split(rng)

            rng, _rng = jax.random.split(rng)
            avail_actions = jax.vmap(env.get_avail_actions)(env_state.log_env_state.env_state)
            avail_actions = jax.lax.stop_gradient(
                batchify(avail_actions, env.agents, eval_config._num_eval_actors)
            )
            obs_batch = batchify(obs, env.agents, eval_config._num_eval_actors)
            ac_in = (
                obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
                avail_actions,
            )

            if eval_config.random_agent:
                action = env.action_space(env_params).sample(rng_act)
            else:
                hstate, pi = actor_network.apply(actor_network_params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                env_act = unbatchify(
                    action, env.agents, eval_config.n_eval_envs, eval_config.n_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

            rng_step = jax.random.split(rng, eval_config.n_eval_envs)
            obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                rng_step, env_state, env_act, env_params
            )
            done_batch = batchify(done, env.agents, eval_config._num_eval_actors).squeeze()
            return (rng, obs, done_batch, env_state, hstate), (env_state, reward, done['__all__'], info)

        print('Scanning episode steps:')
        _, (states, rewards, dones, infos) = jax.lax.scan(
            step_env, (rng, obs, done, env_state, hstate), None,
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

def get_eval_stats(states, dones):
    # Everything has size (n_bins, n_steps, n_envs)
    # Mask out so we only have the final step of each episode
    ep_rews = states.log_env_state.returned_episode_returns * dones[..., None]
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


def get_eval_name(eval_config: MultiAgentEvalConfig, train_config: TrainConfig):
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
    main_eval_ma()