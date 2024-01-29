import copy
import pprint

import hydra
import submitit

from config import EnjoyConfig, EvalConfig, SweepConfig, TrainConfig
from config_sweeps import hypers
from enjoy import main_enjoy
from eval import main_eval
from eval_change_pct import main_eval_cp
from plot import main as main_plot
from train import main as main_train


def get_sweep_cfgs(default_config, hypers):
    # sweep_configs = get_sweep_cfgs(default_config, **hypers)
    sweep_configs = []
    for h in hypers:
        sweep_configs += get_grid_cfgs(default_config, h)
    return sweep_configs


def get_grid_cfgs(default_config, kwargs):
    """Return set of experiment configs corresponding to the grid of 
    hyperparameter values specified by kwargs."""
    subconfigs = [default_config]
    # Name of hyper, list of values
    for k, v in kwargs.items():
        if k == 'NAME':
            continue

        if k == 'obs_size':
            # Break this down into `arf_size` and `vrf_size` with the same value
            assert isinstance(v, list)
            new_subconfigs = []
            for vi in v:
                assert isinstance(vi, int)
                for sc in subconfigs:
                    nsc = copy.deepcopy(sc)
                    setattr(nsc, k, vi)
                    setattr(nsc, 'arf_size', vi)
                    setattr(nsc, 'vrf_size', vi)
                    new_subconfigs.append(nsc)
            subconfigs = new_subconfigs

        elif hasattr(default_config, k):
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

        else:
            raise Exception(f"{k} is not a valid hyperparameter.")
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
    else:
        raise Exception('Invalid mode: f{cfg.mode}')

    # ... but we work around this kind of.
    for k, v in dict(cfg).items():
        setattr(default_config, k, v)

    sweep_configs = get_sweep_cfgs(default_config, hypers)

    # sweep_configs = [(sc,) for sc in sweep_configs]

    if cfg.slurm:

        # Launch rendering sweep on SLURM
        if cfg.mode == 'enjoy':
            executor = submitit.AutoExecutor(folder='submitit_logs')
            executor.update_parameters(
                    job_name=f"{hypers[0]['NAME']}_enjoy",
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
                        job_name=f"{hypers[0]['NAME']}_train",
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
            raise Exception(
                (f'Sweep jobs of mode {cfg.mode} cannot be submitted to SLURM. '
                f'Try again with `slurm=false`.'))

    else:
        return seq_main(main_fn, sweep_configs)

if __name__ == '__main__':
    ret = sweep_main()
