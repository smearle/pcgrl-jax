import copy
import pprint

import hydra
import submitit

from config import EnjoyConfig, EvalConfig, SweepConfig, TrainConfig
from enjoy import main_enjoy
from eval import main_eval
from eval_change_pct import main_eval_cp
from plot import main as main_plot
from train import main as main_train


############ baseline experiments (?) ############
# # (-3) sweeping over action observation window without control
# hypers = {
#     'arf_size': [3, 5, 8, 16, 32],
#     'seed': [0, 1, 2],
#     'model': ['conv', 'seqnca'],
#     'n_envs': [600],
#     'total_timesteps': [200_000_000],
# }

# # (-2) sweeping over value branch observation window without control
# hypers = {
#     'vrf_size': [3, 5, 8, 16, 32],
#     'seed': [0, 1, 2],
#     'model': ['conv', 'seqnca'],
#     'n_envs': [600],
#     'total_timesteps': [200_000_000],
# }

# (-1) sweeping over observation window without control
# hypers = {
#     'obs_size': [3, 5, 8, 16],
#     'seed': [0, 1, 2],
#     'model': ['conv', 'seqnca'],
#     'n_envs': [600],
#     'total_timesteps': [200_000_000],
# }

############ experiments for controllability of different rf sizes ############
# # (2) sweeping over action observation window (formerly "patch width")
# hypers = {
#     'ctrl_metrics': [['diameter']],
#     'arf_size': [3, 5, 8, 16, 32],
#     'seed': [0, 1, 2],
#     'model': ['conv', 'seqnca'],
#     'n_envs': [600],
#     'total_timesteps': [200_000_000],
# }

# # (1) woops, this is actually what we meant at (0)
# hypers = {
#     'ctrl_metrics': [['diameter']],
#     'obs_size': [3, 5, 8, 16],
#     'seed': [0, 1, 2],
#     'model': ['conv', 'seqnca'],
#     'n_envs': [600],
#     'total_timesteps': [200_000_000],
# }

# # (0) first sweep for ICLR
# hypers = {
#     'ctrl_metrics': [['diameter']],
#     'vrf_size': [3, 5, 8, 16, 32],
#     'seed': [0, 1, 2],
#     'model': ['conv', 'seqnca'],
#     'n_envs': [600],
#     'total_timesteps': [200_000_000],
# }



### IJCAI experiments ###

hypers = [
    {
        'change_pct': [0.2, 0.4, 0.6, 0.8, 1.0],
        'seed': [0, 1, 2],
        'n_envs': [600],
        'max_board_scans': [3.0],
        'total_timesteps': [200_000_000],
    },
    {
        'change_pct': [-1.0],
        'seed': [0, 1, 2],
        'n_envs': [600],
        'max_board_scans': [1, 5, 10],
        'total_timesteps': [200_000_000],
    }
]

########################


def get_sweep_cfgs(default_config, kwargs):
    subconfigs = [default_config]
    # Name of hyper, list of values
    for k, v in kwargs.items():
        if hasattr(default_config, k):
            assert isinstance(v, list)
            new_subconfigs = []
            # e.g. different learning rates
            for vi in v:
                for sc in subconfigs:
                    # create a copy of sc
                    nsc = copy.deepcopy(sc)
                    # set the attribute k to vi
                    setattr(nsc, k, vi)
                    new_subconfigs.append(nsc)
            subconfigs = new_subconfigs

        elif k == 'obs_size':
            # Break this down into `arf_size` and `vrf_size` with the same value
            assert isinstance(v, list)
            new_subconfigs = []
            for vi in v:
                assert isinstance(vi, int)
                for sc in subconfigs:
                    nsc = copy.deepcopy(sc)
                    setattr(nsc, 'arf_size', vi)
                    setattr(nsc, 'vrf_size', vi)
                    new_subconfigs.append(nsc)
            subconfigs = new_subconfigs

        else:
            raise Exception
    return subconfigs


def seq_main(main_fn, sweep_configs):
    """Convenience function for executing a sweep of jobs sequentially"""
    return [main_fn(sc) for sc in sweep_configs]


@hydra.main(version_base=None, config_path='./', config_name='batch_pcgrl')
def sweep_main(cfg: SweepConfig):

    # This is a hack. Would mean that we can't overwrite trial-specific settings
    # via hydra yamls or command line arguments...
    if cfg.mode == 'train':
        default_config = TrainConfig()
        main_fn = main_train
    elif cfg.mode == 'plot':
        default_config = TrainConfig()
        main_fn = main_plot
    elif cfg.mode == 'enjoy':
        default_config = EnjoyConfig()
        main_fn = main_enjoy
    elif cfg.mode == 'eval_cp':
        default_config = EvalConfig()
        main_fn = main_eval_cp
    elif cfg.mode == 'eval':
        default_config = EvalConfig()
        main_fn = main_eval

    # ... but we work around this kind of.
    for k, v in dict(cfg).items():
        setattr(default_config, k, v)

    # sweep_configs = get_sweep_cfgs(default_config, **hypers)
    sweep_configs = []
    for h in hypers:
        sweep_configs += get_sweep_cfgs(default_config, h)

    # sweep_configs = [(sc,) for sc in sweep_configs]

    if cfg.slurm:

        # Launch rendering sweep on SLURM
        if cfg.mode == 'enjoy':
            executor = submitit.AutoExecutor(folder='submitit_logs')
            executor.update_parameters(
                    mem_gb=30,
                    tasks_per_node=1,
                    cpus_per_task=1,
                    gpus_per_node=1,
                    timeout_min=60,
                )
            return executor.submit(seq_main, main_enjoy, sweep_configs)

        # TODO: Launch eval sweep on SLURM

        # Launch training sweep on SLURM
        if cfg.mode == 'train':
            if cfg.slurm:
                executor = submitit.AutoExecutor(folder='submitit_logs')
                executor.update_parameters(
                        mem_gb=30,
                        tasks_per_node=1,
                        cpus_per_task=1,
                        # gpus_per_node=1,
                        timeout_min=1440,
                        slurm_gres='gpu:rtx8000:1',
                        # partition='rtx8000',
                    )
                # Pretty print all configs to be executed
                pprint.pprint(sweep_configs)
                executor.map_array(main_fn, sweep_configs)

    else:
        return seq_main(main_fn, sweep_configs)

if __name__ == '__main__':
    sweep_main()
