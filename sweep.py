import copy

import hydra
import submitit

from config import EnjoyConfig, TrainConfig
from enjoy import main_enjoy
from plot import main as main_plot
from train import main as main_train


# hypers = {
#     'static_tile_prob': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
# }


hypers = {
    'representation': ['turtle'],
    'n_agents': [1, 2, 3, 4, 5],
}


def get_sweep_cfgs(default_config, **kwargs):
    subconfigs = [default_config]
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
        else:
            raise Exception
    return subconfigs


def seq_main(main_fn, sweep_configs):
    """Convenience function for executing a sweep of jobs sequentially on a
    SLURM cluster."""
    return [main_fn(sc) for sc in sweep_configs]


@hydra.main(version_base=None, config_path='./', config_name='batch_pcgrl')
def sweep_main(cfg):

    if cfg.mode == 'train':
        main_fn = main_train
        default_config = TrainConfig()
    elif cfg.mode == 'plot':
        main_fn = main_plot
        default_config = TrainConfig()
    elif cfg.mode == 'enjoy':
        default_config = EnjoyConfig()

    for k, v in dict(cfg).items():
        setattr(default_config, k, v)

    sweep_configs = get_sweep_cfgs(default_config, **hypers)
    # sweep_configs = [(sc,) for sc in sweep_configs]

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

    if cfg.slurm:
        executor = submitit.AutoExecutor(folder='submitit_logs')
        executor.update_parameters(
                mem_gb=30,
                tasks_per_node=1,
                cpus_per_task=1,
                gpus_per_node=1,
                timeout_min=1440,
                # partition='rtx8000',
            )
        executor.map_array(main_fn, sweep_configs)
    else:
        [main_fn(sc) for sc in sweep_configs]


if __name__ == '__main__':
    sweep_main()
