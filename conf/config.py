from typing import Iterable, List, Optional, Tuple, Union
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass


# @dataclass
# class EnvConfig:
#     problem: str = "binary"
#     representation: str = "nca"


@dataclass
class Config:
    lr: float = 1.0e-4
    n_envs: int = 4
    num_steps: int = 128
    total_timesteps: int = int(5e7)
    timestep_chunk_size: int = -1
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
    randomize_map_shape: bool = False
    is_3d: bool = False
    # ctrl_metrics: Tuple[str] = ('diameter', 'n_regions')
    ctrl_metrics: Tuple[str] = ()
    # Size of the receptive field to be fed to the action subnetwork.
    vrf_size: Optional[int] = -1  # -1 means 2 * map_width - 1, i.e. full observation, 31 if map_width=16
    # Size of the receptive field to be fed to the value subnetwork.
    arf_size: Optional[int] = -1  # -1 means 2 * map_width - 1, i.e. full observation, 31 if map_width=16
    # TODO: actually take arf and vrf into account in models, where possible

    change_pct: float = -1.0

    # The shape of the (patch of) edit(s) to be made by the edited by the generator at each step.
    act_shape: Tuple[int, int] = (1, 1)

    static_tile_prob: Optional[float] = 0.0
    n_freezies: int = 0
    n_agents: int = 1  # multi-agent is fake and broken
    max_board_scans: float = 3.0

    # How many milliseconds to wait between frames of the rendered gifs
    gif_frame_duration: int = 25

    """ DO NOT USE. WILL BE OVERWRITTEN. """
    exp_dir: Optional[str] = None
    n_gpus: int = 1

    # To make the task simpler, always start with an empty map
    empty_start: bool = False

    # In problems with tile-types with specified valid numbers, fix/freeze their random placement at the beginning of 
    # each episode.
    pinpoints: bool = False

    hidden_dims: Tuple[int] = (64, 256)

    # TODO: Implement this. Just a placeholder for now.
    reward_every: int = 1

    # A toggle, will add `n_envs` to the experiment name if we are profiling training FPS, so that we can distinguish 
    # results.
    profile_fps: bool = False


@dataclass
class EvoMapConfig(Config):
    n_generations: int = 100_000
    evo_pop_size: int = 100
    n_parents: int = 50
    mut_rate: float = 0.3
    render_freq: int = 10_000
    log_freq: int = 1_000
    callbacks: bool = True


@dataclass
class TrainConfig(Config):
    overwrite: bool = False

    # Save a checkpoint after (at least) this many timesteps
    ckpt_freq: int = int(1e7)
    # Render after this many update steps
    render_freq: int = 1000
    n_render_eps: int = 3

    # eval the model on pre-made eval freezie maps to see how it's doing
    eval_freq: int = 100
    n_eval_maps: int = 6
    eval_map_path: str = "user_defined_freezies/binary_eval_maps.json"
    # discount factor for regret value calculation is the same as GAMMA

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    NUM_UPDATES: Optional[int] = None
    MINIBATCH_SIZE: Optional[int] = None
    ###########################################################################


@dataclass
class MultiAgentConfig(TrainConfig):
    representation: str = "turtle"
    n_agents: int = 2
    scale_clip_eps: bool = False
    hidden_dims: Tuple[int] = (256, -1)
    empty_start: bool = True

    # Save a checkpoint after (at least) this many ***update*** steps
    ckpt_freq: int = 1

    # WandB Params
    WANDB_MODE: str = 'run'  # one of: 'offline', 'run', 'dryrun', 'shared', 'disabled', 'online'
    ENTITY: str = ''
    PROJECT: str = 'smearle_pcgrl_mappo'

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    _num_actors: int = -1
    _minibatch_size: int = -1
    _num_updates: int = -1
    _exp_dir: Optional[str] = None
    _ckpt_dir: Optional[str] = None
    _vid_dir: Optional[str] = None
    ###########################################################################


@dataclass
class TrainAccelConfig(TrainConfig):
    evo_freq: int = 10
    evo_pop_size: int = 10
    evo_mutate_prob: float = 0.1


@dataclass
class EvalConfig(TrainConfig):
    reevaluate: bool = True
    random_agent: bool = False
    # In how many bins to divide up each metric being evaluated
    n_bins: int = 10
    n_eval_envs: int = 10
    n_eps: int = 5
    eval_map_width: Optional[int] = None
    eval_max_board_scans: Optional[int] = None
    eval_randomize_map_shape: Optional[bool] = None
    eval_seed: int = 0

    # Which eval metric to keep in our generated table if sweeping over eval hyperparams (in which case we want to 
    # save space). Only applied when runnins `cross_eval.py`
    # metrics_to_keep = [
    #     # 'min_min_loss',
    #     'mean_ep_reward',
    # ]
    metrics_to_keep: Tuple[str] = ('mean_ep_reward',)


@dataclass
class EnjoyConfig(EvalConfig):
    random_agent: bool = False
    # How many episodes to render as gifs
    n_eps: int = 5
    eval_map_width: Optional[int] = None
    render_stats: bool = True
    n_enjoy_envs: int = 1
    render_ims: bool = False


@dataclass
class EnjoyMultiAgentConfig(MultiAgentConfig, EnjoyConfig):
    pass
    

@dataclass
class ProfileEnvConfig(Config):
    N_PROFILE_STEPS: int = 5000
    reevaluate: bool = False


@dataclass
class SweepConfig(EnjoyConfig, EvalConfig):
    name: Optional[str] = None
    mode: str = 'train'
    slurm: bool = True


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="ma_config", node=MultiAgentConfig)
cs.store(name="enjoy_ma_pcgrl", node=EnjoyMultiAgentConfig)
cs.store(name="evo_map_pcgrl", node=EvoMapConfig)
cs.store(name="train_pcgrl", node=TrainConfig)
cs.store(name="train_accel_pcgrl", node=TrainAccelConfig)
cs.store(name="enjoy_pcgrl", node=EnjoyConfig)
cs.store(name="eval_pcgrl", node=EvalConfig)
cs.store(name="profile_pcgrl", node=ProfileEnvConfig)
cs.store(name="batch_pcgrl", node=SweepConfig)
