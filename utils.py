import os

import gymnax

from config import Config
from envs.binary_0 import Binary0
from envs.pcgrl_env import PCGRLEnvParams, PCGRLEnv
from models import ActorCritic, AutoEncoder, ConvForward, Dense, NCA, SeqNCA


def get_exp_dir(config: Config):
    exp_name = os.path.join(
        'saves',
        f'{config.problem}_{config.representation}_{config.model}-' +
        f'{config.activation}_w-{config.maze_size}_rf-{config.rf_size}_' +
        f'arf-{config.arf_size}_sp-{config.static_tile_prob}_' + \
        # f'fz-{config.n_freezies}_' + \
        f'act-{config.act_shape[0]}x{config.act_shape[1]}_' + \
        f'{config.seed}_{config.exp_name}')
    return exp_name


def init_config(config: Config):
    config.rf_size = (2 * config.maze_size -
                      1 if config.rf_size is None else config.rf_size)
    config.arf_size = config.rf_size if config.arf_size is None \
        else config.arf_size
    return config


def get_ckpt_dir(config: Config):
    return os.path.join(get_exp_dir(config), 'ckpts')


def get_network(env: PCGRLEnv, env_params: PCGRLEnvParams, config: Config):
    if config.model == "dense":
        network = Dense(
            env.action_space(env_params).n, activation=config.activation
        )
    if config.model == "conv":
        network = ConvForward(
            env.action_space(env_params).n, activation=config.activation,
            arf_size=config.arf_size,
        )
    if config.model == "seqnca":
        network = SeqNCA(
            env.action_space(env_params).n, activation=config.activation,
            arf_size=config.arf_size,
        )
    if config.model in {"nca", "autoencoder"}:
        if config.representation == "wide":
            action_dim = len(env.tile_enum) - 1
        else:
            action_dim = env.action_space(env_params).n
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
    network = ActorCritic(network, act_shape=config.act_shape)
    return network


def gymnax_pcgrl_make(env_name, config: Config, **env_kwargs):
    if env_name in gymnax.registered_envs:
        return gymnax.make(env_name)

    elif env_name == 'PCGRL':
        map_shape = (config.maze_size, config.maze_size)
        rf_shape = (config.rf_size, config.rf_size)
        env_params = PCGRLEnvParams()
        env = PCGRLEnv(
            problem=config.problem, representation=config.representation,
            map_shape=map_shape, rf_shape=rf_shape,
            static_tile_prob=config.static_tile_prob,
            n_freezies=config.n_freezies, env_params=env_params,
            act_shape=config.act_shape,
        )

    elif env_name == 'Binary0':
        env = Binary0(**env_kwargs)

    return env, env_params
