import os

import gymnax
import jax
import numpy as np

from config import Config, TrainConfig
from envs.binary_0 import Binary0
from envs.candy import Candy, CandyParams
from envs.pcgrl_env import PROB_CLASSES, PCGRLEnvParams, PCGRLEnv, ProbEnum, RepEnum, get_prob_cls
from envs.play_pcgrl_env import PlayPCGRLEnv, PlayPCGRLEnvParams
from envs.probs.binary import BinaryProblem
from envs.probs.problem import Problem
from models import ActorCritic, ActorCriticPCGRL, ActorCriticPlayPCGRL, AutoEncoder, ConvForward, Dense, NCA, SeqNCA


def get_exp_dir(config: Config):
    if config.env_name == 'PCGRL':
        ctrl_str = '_ctrl_' + '_'.join(config.ctrl_metrics) if len(config.ctrl_metrics) > 0 else '' 
        exp_dir = os.path.join(
            'saves',
            f'{config.problem}{ctrl_str}_{config.representation}_{config.model}-' +
            f'{config.activation}_w-{config.map_width}_vrf-{config.vrf_size}_' +
            (f'cp-{config.change_pct}_' if config.change_pct > 0 else '') +
            f'arf-{config.arf_size}_sp-{config.static_tile_prob}_' + \
            f'bs-{config.max_board_scans}_' + \
            f'fz-{config.n_freezies}_' + \
            f'act-{"x".join([str(e) for e in config.act_shape])}_' + \
            f'nag-{config.n_agents}_' + \
            f'{config.seed}_{config.exp_name}')
    elif config.env_name == 'PlayPCGRL':
        exp_dir = os.path.join(
            'saves',
            f'play_w-{config.map_width}_' + \
            f'{config.model}-{config.activation}_' + \
            f'vrf-{config.vrf_size}_arf-{config.arf_size}_' + \
            f'{config.seed}_{config.exp_name}',
        )
    elif config.env_name == 'Candy':
        exp_dir = os.path.join(
            'saves',
            'candy_' + \
            f'{config.seed}_{config.exp_name}',
        )
    else:
        exp_dir = os.path.join(
            'saves',
            config.env_name,
        )
    return exp_dir


def init_config(config: Config, evo=True):
    config.n_gpus = jax.local_device_count()

    if config.env_name == 'Candy':
        config.exp_dir = get_exp_dir(config)
        return config

    config.arf_size = (2 * config.map_width -
                      1 if config.arf_size is None else config.arf_size)
    config.arf_size = config.arf_size if config.arf_size is None \
        else config.arf_size
    config.exp_dir = get_exp_dir(config)
    if hasattr(config, 'evo_pop_size') and hasattr(config, 'n_envs'):
        assert config.n_envs % (config.evo_pop_size * 2) == 0, "n_envs must be divisible by evo_pop_size * 2"
    return config


def get_ckpt_dir(config: Config):
    return os.path.join(get_exp_dir(config), 'ckpts')


def get_network(env: PCGRLEnv, env_params: PCGRLEnvParams, config: Config):
    if config.env_name == 'Candy':
        # In the candy-player environment, action space is flat discrete space over all candy-direction combos.
        action_dim = env.action_space(env_params).n

    elif 'PCGRL' in config.env_name:
        # First consider number of possible tiles
        # action_dim = env.action_space(env_params).n
        action_dim = env.rep.per_tile_action_dim
    
    else:
        action_dim = env.num_actions

    if config.model == "dense":
        network = Dense(
            action_dim, activation=config.activation,
            arf_size=config.arf_size, vrf_size=config.vrf_size,
        )
    if config.model == "conv":
        network = ConvForward(
            action_dim=action_dim, activation=config.activation,
            arf_size=config.arf_size, act_shape=config.act_shape,
            vrf_size=config.vrf_size,
        )
    if config.model == "seqnca":
        network = SeqNCA(
            action_dim, activation=config.activation,
            arf_size=config.arf_size,
            vrf_size=config.vrf_size,
        )
    if config.model in {"nca", "autoencoder"}:
        if config.model == "nca":
            network = NCA(
                representation=config.representation,
                action_dim=action_dim,
                activation=config.activation,
            )
        elif config.model == "autoencoder":
            network = AutoEncoder(
                representation=config.representation,
                action_dim=action_dim,
                activation=config.activation,
            )
    # if config.env_name == 'PCGRL':
    if 'PCGRL' in config.env_name:
        network = ActorCriticPCGRL(network, act_shape=config.act_shape,
                            n_agents=config.n_agents, n_ctrl_metrics=len(config.ctrl_metrics))
    # elif config.env_name == 'PlayPCGRL':
    #     network = ActorCriticPlayPCGRL(network)
    else:
        network = ActorCritic(network)
    return network

        
def get_env_params_from_config(config: Config):
    map_shape = ((config.map_width, config.map_width) if not config.is_3d
                 else (config.map_width, config.map_width, config.map_width))
    rf_size = max(config.arf_size, config.vrf_size)
    rf_shape = (rf_size, rf_size) if not config.is_3d else (rf_size, rf_size, rf_size)

    act_shape = tuple(config.act_shape)
    if config.is_3d:
        assert len(config.act_shape) == 3

    # Convert strings to enum ints
    problem = ProbEnum[config.problem.upper()]
    prob_cls = PROB_CLASSES[problem]
    ctrl_metrics = tuple([int(prob_cls.metrics_enum[c.upper()]) for c in config.ctrl_metrics])

    env_params = PCGRLEnvParams(
        problem=problem,
        representation=int(RepEnum[config.representation.upper()]),
        map_shape=map_shape,
        rf_shape=rf_shape,
        act_shape=act_shape,
        static_tile_prob=config.static_tile_prob,
        n_freezies=config.n_freezies,
        n_agents=config.n_agents,
        max_board_scans=config.max_board_scans,
        ctrl_metrics=ctrl_metrics,
        change_pct=config.change_pct,
    )
    return env_params


def get_play_env_params_from_config(config: Config):
    map_shape = (config.map_width, config.map_width)
    rf_size = max(config.arf_size, config.vrf_size)
    rf_shape = (rf_size, rf_size) if not config.is_3d else (rf_size, rf_size, rf_size)

    return PlayPCGRLEnvParams(
        map_shape=map_shape,
        rf_shape=rf_shape,
    )

def gymnax_pcgrl_make(env_name, config: Config, **env_kwargs):
    if env_name in gymnax.registered_envs:
        return gymnax.make(env_name)

    elif env_name == 'PCGRL':
        env_params = get_env_params_from_config(config)
        env = PCGRLEnv(env_params)

    elif env_name == 'PlayPCGRL':
        env_params = get_play_env_params_from_config(config)
        env = PlayPCGRLEnv(env_params)

    elif env_name == 'Binary0':
        env = Binary0(**env_kwargs)

    elif env_name == 'Candy':
        env_params = CandyParams()
        env = Candy(env_params)

    return env, env_params
