import copy
import json
import os
import pprint

import hydra
from omegaconf import OmegaConf
import submitit
from tqdm import tqdm

from conf.config import EnjoyConfig, EvalConfig, MultiAgentConfig, SweepConfig, TrainConfig
from conf.config_sweeps import eval_hypers
from utils import get_sweep_conf_path, load_sweep_hypers, write_sweep_confs
from enjoy import main_enjoy
from eval import main_eval
from eval_change_pct import main_eval_cp
from plot import main as main_plot
from train import main as main_train
from mappo import main as main_ma_train
from gen_hid_params_per_obs_size import get_hiddims_dict_path 


def am_on_hpc():
    return 'CLUSTER' in os.environ


def get_sweep_cfgs(default_config, hypers, mode, eval_hypers={}):
    # sweep_configs = get_sweep_cfgs(default_config, **hypers)
    sweep_configs = []
    for h in hypers:
        sweep_configs += get_grid_cfgs(default_config, h, mode, eval_hypers)
    return sweep_configs

    
def get_hiddims_dict(hiddims_dict_path):
    hid_params = json.load(open(hiddims_dict_path, 'r'))
    # Turn the dictionary of (obs_size, hid_dims) to dict with obs_size as key
    hid_dims_dict = {}
    for model_obs_size, hid_dims, n_params in hid_params:
        # Should we be using this model info here?
        model, obs_size = model_obs_size
        hid_dims_dict[obs_size] = hid_dims
    return hid_dims_dict


def get_grid_cfgs(base_config, hypers, mode, eval_hypers={}):
    """Return set of experiment configs corresponding to the grid of 
    hyperparameter values specified by kwargs."""

    # If models were trained with different max_board_scans, evaluate them on the highest such value, for fairness.
    if 'eval' in mode or 'enjoy' in mode:
        if 'max_board_scans' in hypers.keys() and 'max_board_scans' not in eval_hypers:
            base_config.eval_max_board_scans = max(hypers['max_board_scans'])

        # Add eval hypers
        hypers = {**hypers, **eval_hypers}

    # Because this may depend on a bunch of other hyperparameters, so we need to compute hiddims last.
    has_obs_size_hid_dims = False
    if 'obs_size_hid_dims' in hypers:
        has_obs_size_hid_dims = True
        obs_size_hid_dims = hypers.pop('obs_size_hid_dims')
    items = sorted(list(hypers.items()))
    if has_obs_size_hid_dims:
        items.append(('obs_size_hid_dims', obs_size_hid_dims))

    subconfigs = [base_config]
    # Name of hyper, list of values
    hid_dims_dicts = {}
    for k, v in items:
        if k == 'NAME' or k == 'multiagent':
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
    # return [main_fn(sc) for sc in sweep_configs]
    results = []
    for sc in tqdm(sweep_configs):
        results.append(main_fn(sc))
    return results


@hydra.main(version_base=None, config_path='./', config_name='batch_pcgrl')
def sweep_main(cfg: SweepConfig):
    if cfg.mode == 'plot' or not am_on_hpc():
        cfg.slurm = False

    if cfg.name is not None:
        _hypers, _eval_hypers = load_sweep_hypers(cfg)
        _hypers = [_hypers]
        sweep_name = cfg.name
    else:
        from conf.config_sweeps import hypers
        _hypers = hypers
        _eval_hypers = eval_hypers
        write_sweep_confs(_hypers, _eval_hypers)

        sweep_name = _hypers[0]['NAME']
    
    cfg.multiagent = _hypers[0]['multiagent']


    # This is a hack. Would mean that we can't overwrite trial-specific settings
    # via hydra yamls or command line arguments...
    if cfg.mode == 'train':
        # if cfg.n_agents > 1:
        if cfg.multiagent:
            default_config = MultiAgentConfig()
            # default_config = OmegaConf.create(default_config)
            main_fn = main_ma_train
        else:
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
    # elif cfg.mode == 'eval_diff_size':
    #     default_config = EvalConfig()
    #     main_fn = main_eval_diff_size
    elif cfg.mode == 'eval':
        default_config = EvalConfig()
        main_fn = main_eval
    else:
        raise Exception('Invalid mode: f{cfg.mode}')

    # ... but we work around this kind of.
    for k, v in dict(cfg).items():
        setattr(default_config, k, v)

    sweep_configs = get_sweep_cfgs(default_config, _hypers, mode=cfg.mode, eval_hypers=_eval_hypers)
    sweep_configs = [OmegaConf.create(sc) for sc in sweep_configs]

    # sweep_configs = [(sc,) for sc in sweep_configs]

    if cfg.slurm:

        # Launch rendering sweep on SLURM
        if cfg.mode == 'enjoy':
            executor = submitit.AutoExecutor(folder='submitit_logs')
            executor.update_parameters(
                    job_name=f"{sweep_name}_enjoy",
                    mem_gb=90,
                    tasks_per_node=1,
                    cpus_per_task=1,
                    gpus_per_node=1,
                    timeout_min=60,
                    slurm_account='pr_174_general',
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
                    slurm_account='pr_174_general',
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
                    slurm_account='pr_174_general',
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
