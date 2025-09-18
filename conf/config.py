from typing import Iterable, List, Optional, Tuple, Union
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass


# @dataclass
# class EnvConfig:
#     problem: str = "binary"
#     representation: str = "nca"


@dataclass
class Config:
    save_dir: str = "saves"
    lr: float = 1.0e-4
    n_envs: int = 400
    # How many steps do I take in all of my batched environments before doing a gradient update
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
    n_agents: int = 1
    multiagent: bool = False
    max_board_scans: float = 3.0

    # How many milliseconds to wait between frames of the rendered gifs
    gif_frame_duration: int = 25

    # To make the task simpler, always start with an empty map
    empty_start: bool = False
    # Or a full (all-wall) map
    full_start: bool = False

    # In problems with tile-types with specified valid numbers, fix/freeze their random placement at the beginning of 
    # each episode.
    pinpoints: bool = False

    hidden_dims: Tuple[int] = (256, 512)

    # A toggle, will add `n_envs` to the experiment name if we are profiling training FPS, so that we can distinguish 
    # results.
    profile_fps: bool = False

    reward_freq: int = 1


    """ DO NOT USE. WILL BE OVERWRITTEN. """
    exp_dir: Optional[str] = None
    n_gpus: int = 1
    _is_recurrent: bool = False


@dataclass
class EvoMapConfig(Config):
    save_dir: str = "saves_evo_map"
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
    render_freq: int = 20
    n_render_eps: int = 3

    # eval the model on pre-made eval freezie maps to see how it's doing
    eval_freq: int = 100
    n_eval_maps: int = 6
    eval_map_path: str = "user_defined_freezies/binary_eval_maps.json"
    # discount factor for regret value calculation is the same as GAMMA

    # WandB Params
    wandb_mode: str = 'run'  # one of: 'offline', 'run', 'dryrun', 'shared', 'disabled', 'online'
    wandb_entity: str = ''
    wandb_project: str = 'smearle_pcgrl_mappo'


    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    obs_size_hid_dims: int = -1
    _num_updates: int = -1
    _minibatch_size: int = -1
    ###########################################################################


@dataclass
class MultiAgentConfig(TrainConfig):
    multiagent: bool = True
    # lr: float = 3e-4
    # update_epochs: int = 4
    # num_steps: int = 521
    # gamma: float = 0.99
    # gae_lambda: float = 0.95
    # clip_eps: float = 0.2
    # scale_clip_eps: bool = False
    # ent_coef: float = 0.0
    # vf_coef: float = 0.5
    # max_grad_norm: float = 0.25

    model: str = 'rnn'
    representation: str = "turtle"
    n_agents: int = 2
    n_envs: int = 400
    n_eval_envs: int = 10
    scale_clip_eps: bool = False
    # hidden_dims: Tuple[int] = (512, 256)
    a_freezer: bool = False
    per_agent_reward_freq: int = -1

    # Save a checkpoint after (at least) this many ***update*** steps
    ckpt_freq: int = 40
    render_freq: int = 20

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    _num_actors: int = -1
    _num_eval_actors: int = -1
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
    n_eps: int = 5
    eval_map_width: Optional[int] = None
    eval_max_board_scans: Optional[float] = None
    eval_randomize_map_shape: Optional[bool] = None
    eval_seed: int = 0

    # Which eval metric to keep in our generated table if sweeping over eval hyperparams (in which case we want to 
    # save space). Only applied when running `cross_eval.py`
    # metrics_to_keep = [
    #     # 'min_min_loss',
    #     'mean_ep_reward',
    # ]

    metrics_to_keep: Tuple[str] = ('mean_ep_reward',)
    # metrics_to_keep: Tuple[str] = ('mean_fps',)


@dataclass
class EvalMultiAgentConfig(MultiAgentConfig, EvalConfig):
    pass


@dataclass
class EnjoyConfig(EvalConfig):
    random_agent: bool = False
    # How many episodes to render as gifs
    n_eps: int = 5
    eval_map_width: Optional[int] = None
    # Add debugging text showing the current/target values for various stats to each frame of the episode (this is really slow)
    render_stats: bool = False
    n_enjoy_envs: int = 1
    render_ims: bool = False
    a_freezer: bool = False


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
    overwrite: bool = False
    n_eval_envs: int = 10

@dataclass
class GetTracesConfig(EnjoyConfig):
    len_trace: int = 3 # deprecated, will filter the dataset later, just leave it here in case
    n_enjoy_envs: int = 1
    n_eps: int = 1  # keep it at 1 for each trace
    render_stats: bool = False
    render_ims: bool = False
    overwrite_traces: bool = False

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="ma_config", node=MultiAgentConfig)
cs.store(name="enjoy_ma_pcgrl", node=EnjoyMultiAgentConfig)
cs.store(name="evo_map_pcgrl", node=EvoMapConfig)
cs.store(name="train_pcgrl", node=TrainConfig)
cs.store(name="train_accel_pcgrl", node=TrainAccelConfig)
cs.store(name="enjoy_pcgrl", node=EnjoyConfig)
cs.store(name="eval_pcgrl", node=EvalConfig)
cs.store(name="eval_ma_pcgrl", node=EvalMultiAgentConfig)
cs.store(name="profile_pcgrl", node=ProfileEnvConfig)
cs.store(name="batch_pcgrl", node=SweepConfig)
cs.store(name="get_traces_pcgrl", node=GetTracesConfig)
