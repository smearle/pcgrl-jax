import dataclasses
from functools import partial
import math
import os
import pickle
import shutil
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict

import chex
import distrax
import imageio
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.training import orbax_utils
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
import functools
from flax.training.train_state import TrainState
import hydra
from omegaconf import DictConfig, OmegaConf
from time import perf_counter

from marl.environments.spaces import Box
from marl.environments.multi_agent_env import MultiAgentEnv
from marl.model import ActorCategorical, ActorMLP, ActorRNN, CriticRNN, ScannedRNN

from conf.config import MultiAgentConfig
from envs.pcgrl_env import PCGRLEnv, PCGRLEnvState
from marl.wrappers.baselines import MALogWrapper, MultiAgentWrapper
from utils import get_env_params_from_config, get_exp_dir, init_config


@struct.dataclass
class RunnerState:
    train_states: Tuple[TrainState, TrainState]
    env_state: MultiAgentEnv
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    hstates: Tuple[jnp.ndarray, jnp.ndarray]
    rng: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    #print('batchify', x.shape)
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def linear_schedule(config, count):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["LR"] * frac


def init_run(config: MultiAgentConfig, ckpt_manager, latest_update_step, rng):
    # Create PCGRL environment
    env_params = get_env_params_from_config(config)
    env = PCGRLEnv(env_params)

    # Wrap environment with JAXMARL wrapper
    env = MultiAgentWrapper(env, env_params)

    # Wrap environment with LogWrapper
    env = MALogWrapper(env)

    # Configure training
    config._num_actors = env.n_agents * config.n_envs
    
    config._num_updates = int(
        config.total_timesteps // config.n_envs // config.n_envs
    )
    config._minibatch_size = (
        config._num_actors * config.num_steps // config.NUM_MINIBATCHES
    )
    config.CLIP_EPS = (
        config.CLIP_EPS / env.n_agents
        if config.scale_clip_eps
        else config.CLIP_EPS
    )
 
    actor_network = ActorCategorical(env.action_space(env.agents[0]).n,
                             subnet=ActorRNN(env.action_space(env.agents[0]).n, config=config,
                            #  subnet=ActorMLP(env.action_space(env.agents[0]).shape[0], config=config,
                                             ))
    critic_network = CriticRNN(config=config)
    rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
    # ac_init_x = (
    #     jnp.zeros((1, config.n_envs, env.observation_space(env.agents[0]).shape[0])),
    #     jnp.zeros((1, config.n_envs)),
    #     jnp.zeros((1, config.n_envs, env.action_space(env.agents[0]).n)),
    # )
    # ac_init_hstate = ScannedRNN.initialize_carry(config.n_envs, config.hidden_dims[0])
    ac_init_x, ac_init_hstate = env.gen_dummy_obs(config)
    actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)

    print(actor_network.subnet.tabulate(rngs=_rng_actor, x=ac_init_x, hidden=ac_init_hstate))

    cr_init_x = (
        jnp.zeros((1, config.n_envs, env.world_state_size,)),  
        jnp.zeros((1, config.n_envs)),
    )
    cr_init_hstate = ScannedRNN.initialize_carry(config.n_envs, config.hidden_dims[0])
    critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

    _linear_schedule = partial(linear_schedule, config)
    
    if config.ANNEAL_LR:
        actor_tx = optax.chain(
            optax.clip_by_global_norm(config.MAX_GRAD_NORM),
            optax.adam(learning_rate=_linear_schedule, eps=1e-5),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config.MAX_GRAD_NORM),
            optax.adam(learning_rate=_linear_schedule, eps=1e-5),
        )
    else:
        actor_tx = optax.chain(
            optax.clip_by_global_norm(config.MAX_GRAD_NORM),
            optax.adam(config.lr, eps=1e-5),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config.MAX_GRAD_NORM),
            optax.adam(config.lr, eps=1e-5),
        )
    actor_train_state = TrainState.create(
        apply_fn=actor_network.apply,
        params=actor_network_params,
        tx=actor_tx,
    )
    critic_train_state = TrainState.create(
        apply_fn=actor_network.apply,
        params=critic_network_params,
        tx=critic_tx,
    )

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.n_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
    ac_init_hstate = ScannedRNN.initialize_carry(config._num_actors, config.hidden_dims[0])
    cr_init_hstate = ScannedRNN.initialize_carry(config._num_actors, config.hidden_dims[0])

    rng, _rng = jax.random.split(rng)
    runner_state = RunnerState(
        (actor_train_state, critic_train_state),
        env_state,
        obsv,
        jnp.zeros((config._num_actors), dtype=bool, ),
        (ac_init_hstate, cr_init_hstate),
        _rng,
    )

    return runner_state, actor_network, env, latest_update_step


def restore_run(config: MultiAgentConfig, runner_state: RunnerState, ckpt_manager, latest_update_step: int):
    if latest_update_step is not None:
        runner_state = ckpt_manager.restore(latest_update_step, items=runner_state)
        with open(os.path.join(config._exp_dir, "wandb_run_id.txt"), "r") as f:
            wandb_run_id = f.read()
    else:
        wandb_run_id=None

    return runner_state, wandb_run_id


def make_sim_render_episode(config: MultiAgentConfig, actor_network, env: PCGRLEnv):
    
    # FIXME: Shouldn't hardcode this
    max_episode_len = env.max_steps
    
    # remaining_timesteps = init_state.env_state.remaining_timesteps
    # actor_params = runner_state.train_states[0].params
    # actor_hidden = runner_state.hstates[0]

    def sim_render_episode(actor_params, actor_hidden):
        rng = jax.random.PRNGKey(0)
        
        init_obs, init_state = env.reset(rng)
        init_obs = batchify(init_obs, env.agents, env.n_agents) 
        
        def step_env(carry, _):
            rng, obs, state, done, actor_hidden = carry
            # print(obs.shape)

            # traj = datatypes.dynamic_index(
            #     state.env_state.sim_trajectory, state.env_state.timestep, axis=-1, keepdims=True
            # )
            avail_actions = env.get_avail_actions(state.env_state)
            avail_actions = jax.lax.stop_gradient(
                batchify(avail_actions, env.agents, len(env.agents))
            )
            ac_in = (
                obs[np.newaxis, :],
                # obs,
                done[np.newaxis, :],
                # done,
                avail_actions[np.newaxis, :],
            )
            actor_hidden, pi = actor_network.apply(actor_params, actor_hidden, ac_in)            
            action = pi.sample(seed=rng)
            env_act = unbatchify(
                action, env.agents, 1, env.n_agents
            )
            env_act = {k: v.squeeze() for k, v in env_act.items()}

            # outputs = [
            #     jit_select_action({}, state, obs, None, rng)
            #     for jit_select_action in jit_select_action_list
            # ]
            # action = agents.merge_actions(outputs)
            obs, next_state, reward, done, info = env.step(state=state, action=env_act, key=rng)
            rng, _ = jax.random.split(rng)
            done = batchify(done, env.agents, env.n_agents)[:, 0]
            obs = batchify(obs, env.agents, env.n_agents)

            return (rng, obs, next_state, done, actor_hidden), next_state

            
        done = jnp.zeros((len(env.agents),), dtype=bool)

        _, states = jax.lax.scan(step_env, (rng, init_obs, init_state, done, actor_hidden), None, length=max_episode_len)

        # Concatenate the init_state to the states
        states = jax.tree.map(lambda x, y: jnp.concatenate([x[None], y], axis=0), init_state, states)

        frames = jax.vmap(env.render)(states.env_state)

        return frames

    return jax.jit(sim_render_episode)

# states = []
# rng, obs, state, done, actor_hidden = (rng, init_obs, init_state, done, actor_hidden)
# for i in range(remaining_timesteps):
#     carry, state = step_env((rng, obs, state, done, actor_hidden), None)
#     rng, obs, state, done, actor_hidden = carry
#     states.append(state)

    
def render_callback(env: PCGRLEnv, frames, save_dir: str, t: int, max_steps: int):

    imageio.mimsave(os.path.join(save_dir, f"enjoy_{t}.gif"), np.array(frames), fps=20, loop=0)
    wandb.log({"video": wandb.Video(os.path.join(save_dir, f"enjoy_{t}.gif"), fps=20, format="gif")})


def get_ckpt_dir(config: MultiAgentConfig):
    ckpts_dir = os.path.abspath(os.path.join(config._exp_dir, "ckpts"))
    return ckpts_dir

    
def ma_init_config(config: MultiAgentConfig):
    config._exp_dir = get_exp_dir(config)
    config._ckpt_dir = get_ckpt_dir(config)
    config._vid_dir = os.path.join(config._exp_dir, "vids")
    init_config(config)

    
def save_checkpoint(config: MultiAgentConfig, ckpt_manager, runner_state, t):
    save_args = orbax_utils.save_args_from_target(runner_state)
    ckpt_manager.save(t.item(), runner_state, save_kwargs={'save_args': save_args})
    ckpt_manager.wait_until_finished() 