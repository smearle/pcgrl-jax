import copy

import submitit

from config import TrainConfig
from train import main


hypers = {
    'static_tile_prob': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    # 'n_agents': [1, 2, 3, 4, 5, 6, 7, 8],
}

def main():
    default_config = TrainConfig(
            total_timesteps=1_000_000_000,
        )

    sweep_configs = get_sweep_cfgs(default_config, **hypers)
    # sweep_configs = [(sc,) for sc in sweep_configs]

    executor = submitit.AutoExecutor(folder='submitit_logs')
    executor.update_parameters(
            mem_gb=30,
            tasks_per_node=1,
            cpus_per_task=1,
            gpus_per_node=1,
            timeout_min=1440,
            )
    executor.map_array(main, sweep_configs)


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
        else:
            raise Exception
    return new_subconfigs


if __name__ == '__main__':
    main()
