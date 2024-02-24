import copy
import json
import os
import pprint

import hydra
import submitit

from config import EnjoyConfig, EvalConfig, SweepConfig, TrainConfig
from config_sweeps import hypers
from utils import load_sweep_hypers
from enjoy import main_enjoy
from eval import main_eval
from eval_change_pct import main_eval_cp
from eval_different_size import main_eval_diff_size
from plot import main as main_plot
from train import main as main_train
from gen_hid_params_per_obs_size import get_hiddims_dict_path 


def get_sweep_cfgs(default_config, hypers):
    # sweep_configs = get_sweep_cfgs(default_config, **hypers)
    sweep_configs = []
    for h in hypers:
        sweep_configs += get_grid_cfgs(default_config, h)
    return sweep_configs

    
def get_hiddims_dict(hiddims_dict_path):
    hid_params = json.load(open(hiddims_dict_path, 'r'))
    # Turn the dictionary of (obs_size, hid_dims) to dict with obs_size as key
    hid_dims_dict = {}
    for obs_size, hid_dims, n_params in hid_params:
        hid_dims_dict[obs_size] = hid_dims
    return hid_dims_dict


def get_grid_cfgs(base_config, kwargs):
    """Return set of experiment configs corresponding to the grid of 
    hyperparameter values specified by kwargs."""

    # Because this may depend on a bunch of other hyperparameters, so we need to compute hiddims last.
    if 'obs_size_hid_dims' in kwargs:
        obs_size_hid_dims = kwargs.pop('obs_size_hid_dims')
    items = sorted(list(kwargs.items()))
    items.append(('obs_size_hid_dims', obs_size_hid_dims))

    subconfigs = [base_config]
    # Name of hyper, list of values
    hid_dims_dicts = {}
    for k, v in items:
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

        elif k == 'obs_size_hid_dims':

            hid_dims_base_dict_path = get_hiddims_dict_path(base_config)
            if os.path.isfile(hid_dims_base_dict_path):
                hid_dims_dict = get_hiddims_dict(hid_dims_base_dict_path)
                hid_dims_dicts[hid_dims_base_dict_path] = hid_dims_dict

            # Break this down into `arf_size` and `vrf_size` with the same value
            assert isinstance(v, list)
            new_subconfigs = []
            for vi in v:
                obs_size = vi
                assert isinstance(obs_size, int)
                for sc in subconfigs:

                    hid_dims_sc_dict_path = get_hiddims_dict_path(sc)
                    if hid_dims_sc_dict_path not in hid_dims_dicts:
                        hid_dims_dict = get_hiddims_dict(hid_dims_sc_dict_path)
                        hid_dims_dicts[hid_dims_sc_dict_path] = hid_dims_dict

                    # print(f"obs_size {obs_size}")
                    if obs_size == -1:
                        obs_size_d = base_config.map_width * 2 - 1
                    else:
                        obs_size_d = obs_size
                    hidden_dims = hid_dims_dict[obs_size_d]
                    # print(f"hidden_dims {hidden_dims}")

                    nsc = copy.deepcopy(sc)
                    setattr(nsc, k, vi)
                    setattr(nsc, 'arf_size', obs_size)
                    setattr(nsc, 'vrf_size', obs_size)
                    setattr(nsc, 'hidden_dims', hidden_dims)
                    new_subconfigs.append(nsc)
            subconfigs = new_subconfigs

        elif hasattr(base_config, k):
            assert isinstance(v, list)
            new_subconfigs = []
            # e.g. different learning rates
            for vi in v:
                for sc in subconfigs:
                    # create a copy of sc
                    nsc = copy.deepcopy(sc)
                    # set the attribute k to vi
                    setattr(nsc, k, vi)

                    # filter out the invalid combinations
                    if max([nsc.arf_size, nsc.vrf_size]) >= nsc.map_width * 2 - 1:
                        print(f"arf_size {nsc.arf_size} or vrf_size {nsc.vrf_size} too big for map size {nsc.map_width}, not necessary, skipping this config.")
                        # Note: assuming we already run arf/vrf_size == -1, so we can skip this (>=) case
                        continue
                    if nsc.model == 'conv2' and nsc.arf_size != nsc.vrf_size:
                        print(f"arf_size {nsc.arf_size} and vrf_size {nsc.vrf_size} must be equal for conv2 model, skipping this config.")
                        continue

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
    cfg.slurm = False if cfg.mode == 'plot' else cfg.slurm

    if cfg.name is not None:
        _hypers = [load_sweep_hypers(cfg)]
        sweep_name = cfg.name
    else:
        _hypers = hypers
        sweep_name = _hypers[0]['NAME']

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
    elif cfg.mode == 'eval_diff_size':
        default_config = EvalConfig()
        main_fn = main_eval_diff_size
    elif cfg.mode == 'eval':
        default_config = EvalConfig()
        main_fn = main_eval
    else:
        raise Exception('Invalid mode: f{cfg.mode}')

    # ... but we work around this kind of.
    for k, v in dict(cfg).items():
        setattr(default_config, k, v)

    sweep_configs = get_sweep_cfgs(default_config, _hypers)

    # sweep_configs = [(sc,) for sc in sweep_configs]

    if cfg.slurm:

        # Launch rendering sweep on SLURM
        if cfg.mode == 'enjoy':
            executor = submitit.AutoExecutor(folder='submitit_logs')
            executor.update_parameters(
                    job_name=f"{sweep_name}_enjoy",
                    mem_gb=30,
                    tasks_per_node=1,
                    cpus_per_task=1,
                    gpus_per_node=1,
                    timeout_min=60,
                )
            return executor.submit(seq_main, main_fn, sweep_configs)

        # Launch eval sweep on SLURM
        elif cfg.mode.startswith('eval'):
            executor = submitit.AutoExecutor(folder='submitit_logs')
            executor.update_parameters(
                    slurm_job_name=f"eval_{sweep_name}",
                    mem_gb=30,
                    tasks_per_node=1,
                    cpus_per_task=1,
                    timeout_min=60,
                    # gpus_per_node=1,
                    slurm_gres='gpu:rtx8000:1',
                )
            pprint.pprint(sweep_configs)
            return executor.map_array(main_fn, sweep_configs)

        # Launch training sweep on SLURM
        elif cfg.mode == 'train':
            executor = submitit.AutoExecutor(folder='submitit_logs')
            executor.update_parameters(
                    job_name=f"{sweep_name}_train",
                    mem_gb=30,
                    tasks_per_node=1,
                    cpus_per_task=1,
                    timeout_min=1440,
                    # gpus_per_node=1,
                    slurm_gres='gpu:rtx8000:1',
                    # partition='rtx8000',
                )
            # Pretty print all configs to be executed
            pprint.pprint(sweep_configs)
            return executor.map_array(main_fn, sweep_configs)

        else:
            raise Exception(
                (f'Sweep jobs of mode {cfg.mode} cannot be submitted to SLURM. '
                f'Try again with `slurm=false`.'))

    else:
        return seq_main(main_fn, sweep_configs)

if __name__ == '__main__':
    ret = sweep_main()
