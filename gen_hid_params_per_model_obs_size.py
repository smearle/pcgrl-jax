import copy
import json
import os

import hydra
import imageio
import jax
from jax import numpy as jnp
import numpy as np

from conf.config import EnjoyConfig
from envs.pcgrl_env import PCGRLEnv
from marl.wrappers.baselines import MultiAgentWrapper
from utils import init_network, gymnax_pcgrl_make, init_config


models = [
    # 'seqnca', 
    # 'conv2',
    # 'conv'
    'rnn'
]

def compute_n_params(config):
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    network = init_network(env, env_params, config)
    rng = jax.random.PRNGKey(42)
    if config.model == 'rnn':
        # FIXME: Hack
        env = MultiAgentWrapper(env, env_params)
        init_x, init_hstate = env.gen_dummy_obs(config)
        network_params = network.init(rng, init_hstate, init_x)
    else:
        init_x = env.gen_dummy_obs(env_params)
        network_params = network.init(rng, init_x)
    _rng = jax.random.PRNGKey(42)
    # print(network.subnet.tabulate(_rng, init_x.map_obs, init_x.flat_obs))

    
    n_parameters = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(network_params) if isinstance(p, jnp.ndarray))
    return n_parameters


@hydra.main(version_base=None, config_path='./', config_name='enjoy_pcgrl')
def main_gen_hid(config: EnjoyConfig):
    config = init_config(config)
    env: PCGRLEnv


    # Assuming arf=vrf for now
    obs_size = config.arf_size = config.vrf_size = 2 * config.map_width - 1

    obss_model_params = []
    models_base_n_params = []
    for model in models:
        config.model = model
        model_base_n_params = int(compute_n_params(config))
        models_base_n_params.append(model_base_n_params)
    base_n_params = max(models_base_n_params)

    base_hid_dims = copy.copy(config.hidden_dims)
    base_obs_size = obs_size

    # Something is cursed about the way we copy and mutate configs here
    for model in models:
        config.arf_size = config.vrf_size = base_obs_size
        config.model = model
        config.hidden_dims = base_hid_dims
        for obs_size in list(range(3, config.arf_size+1))[::-1]:
            config.arf_size = config.vrf_size = obs_size
            n_params = compute_n_params(config)
            while n_params <= base_n_params:
                obss_model_params.append(((model, obs_size), tuple([int(hd) for hd in config.hidden_dims]), int(n_params)))
                if len(config.hidden_dims) > 1:
                    if config.hidden_dims[0] == config.hidden_dims[1]:
                        config.hidden_dims[1] += 1
                    else:
                        config.hidden_dims[0] += 1
                else:
                    config.hidden_dims[0] += 1
                n_params = compute_n_params(config)
                print(f"model {model} obs_size {obs_size} hidden_dims {config.hidden_dims} n_params {n_params}")

            # Save as json
            hid_params_path = get_hiddims_dict_path(config)

            with open(hid_params_path, 'w') as f:
                json.dump(obss_model_params, f)

def get_hiddims_dict_path(config):
        hid_params_path = os.path.join('conf',
            f"{config.problem}_{config.representation}_{'_'.join(models)}_w-{config.map_width}_hid_params.json")
        return hid_params_path


if __name__ == '__main__':
    main_gen_hid()
