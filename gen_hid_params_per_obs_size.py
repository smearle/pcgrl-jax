import copy
import json
import os

import hydra
import imageio
import jax
from jax import numpy as jnp
import numpy as np

from config import EnjoyConfig
from envs.pcgrl_env import PCGRLEnv, render_stats, gen_dummy_queued_state
from train import init_checkpointer
from utils import get_exp_dir, init_network, gymnax_pcgrl_make, init_config


@hydra.main(version_base=None, config_path='./', config_name='enjoy_pcgrl')
def main_gen_hid(config: EnjoyConfig):
    config = init_config(config)
    env: PCGRLEnv

    def compute_n_params(config):
        env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
        network = init_network(env, env_params, config)
        rng = jax.random.PRNGKey(42)
        init_x = env.gen_dummy_obs(env_params)
        network_params = network.init(rng, init_x)
        n_parameters = sum(np.prod(p.shape) for p in jax.tree_leaves(network_params) if isinstance(p, jnp.ndarray))
        return n_parameters

    # Assuming arf=vrf for now
    obs_size = config.arf_size = config.vrf_size = 2 * config.map_width - 1
    base_n_params = int(compute_n_params(config))
    obs_size_to_params = [
        (obs_size, tuple([int(hd) for hd in config.hidden_dims]), base_n_params)
    ]
    for obs_size in list(range(3, config.arf_size))[::-1]:
        config.arf_size = config.vrf_size = obs_size
        n_params = compute_n_params(config)
        new_n_params = n_params
        new_config = copy.deepcopy(config)
        while new_n_params <= base_n_params:
            config = copy.deepcopy(new_config)
            n_params = new_n_params
            if len(new_config.hidden_dims) > 1:
                if new_config.hidden_dims[0] == new_config.hidden_dims[1]:
                    new_config.hidden_dims[1] += 1
                else:
                    new_config.hidden_dims[0] += 1
            else:
                new_config.hidden_dims[0] += 1
            new_n_params = compute_n_params(new_config)
            print(f"obs_size {obs_size} hidden_dims {new_config.hidden_dims} n_params {new_n_params}")
        obs_size_to_params.append((obs_size, tuple([int(hd) for hd in config.hidden_dims]), int(n_params)))

        # Save as json
        hid_params_path = get_hiddims_dict_path(config)

        with open(hid_params_path, 'w') as f:
            json.dump(obs_size_to_params, f)

def get_hiddims_dict_path(config):
        hid_params_path = os.path.join('conf',
            f"{config.problem}_{config.representation}_{config.model}_w-{config.map_width}_hid_params.json")
        return hid_params_path


if __name__ == '__main__':
    main_gen_hid()
