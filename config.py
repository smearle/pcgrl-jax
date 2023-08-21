from typing import Optional, Tuple
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass


# @dataclass
# class EnvConfig:
#     problem: str = "binary"
#     representation: str = "nca"


@dataclass
class Config:
    lr: float = 1.0e-4
    num_envs: int = 4
    num_steps: int = 128
    total_timesteps: int = int(5e7)
    update_epochs: int = 10
    NUM_MINIBATCHES: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    activation: str = "relu"
    env_name: str = "PCGRL"
    ANNEAL_LR: bool = False
    DEBUG: bool = True
    exp_name: str = "0"
    seed: int = 0

    problem: str = "binary"
    representation: str = "narrow"
    model: str = "conv"

    map_width: int = 16
    rf_size: Optional[int] = None
    arf_size: Optional[int] = None
    act_shape: Tuple[int, int] = (1, 1)
    static_tile_prob: Optional[float] = 0.0
    n_freezies: int = 0
    n_agents: int = 1


@dataclass
class TrainConfig(Config):
    overwrite: bool = False

    # Save a checkpoint after (at least) this many timesteps
    ckpt_freq: int = int(1e6)

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    NUM_UPDATES: Optional[int] = None
    MINIBATCH_SIZE: Optional[int] = None
    ###########################################################################


@dataclass
class EnjoyConfig(Config):
    random_agent: bool = False
    # How many episodes to render as gifs
    n_eps: int = 10
    # How many milliseconds to wait between frames of the rendered gifs
    duration: int = 50


@dataclass
class ProfileEnvConfig(Config):
    N_PROFILE_STEPS: int = 5000


@dataclass
class BatchConfig(TrainConfig):
    mode: str = 'train'
    slurm: bool = True


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="train_pcgrl", node=TrainConfig)
cs.store(name="enjoy_pcgrl", node=EnjoyConfig)
cs.store(name="profile_pcgrl", node=ProfileEnvConfig)
cs.store(name="batch_pcgrl", node=BatchConfig)
