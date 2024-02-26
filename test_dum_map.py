import os

import hydra
import imageio
import jax
import jax.numpy as jnp
import numpy as np

from conf.config import EnjoyConfig
from envs.pathfinding import get_path_coords_diam, get_max_path_length
from envs.pcgrl_env import PCGRLEnvState, render_stats
from envs.probs.maze import MazeTiles
from train import init_checkpointer
from utils import get_exp_dir, init_network, gymnax_pcgrl_make, init_config


@hydra.main(version_base=None, config_path='./', config_name='enjoy_pcgrl')
def main_test_dum_map(config: EnjoyConfig):
    config.problem = 'maze'
    config = init_config(config)

    exp_dir = config.exp_dir
    if not config.random_agent:
        checkpoint_manager, restored_ckpt = init_checkpointer(config)
        network_params = restored_ckpt['runner_state'].train_state.params
    elif not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env.prob.init_graphics()
    network = init_network(env, env_params, config)

    rng = jax.random.PRNGKey(42)

    env_state: PCGRLEnvState
    obs, env_state = env.reset(rng, env_params)
    ep_i = 0

    width, height = env.map_shape
    dummy_env_map = [[MazeTiles.EMPTY] * width] * height,
    
    dummy_env_map = np.array(dummy_env_map)
    dummy_env_map = dummy_env_map[0]
    dummy_env_map[0,0] = MazeTiles.PLAYER
    dummy_env_map[-1,-5] = MazeTiles.DOOR
    dummy_env_map = jnp.array(dummy_env_map)
    prob_state = env.prob.get_curr_stats(dummy_env_map)
    env_state = env_state.replace(prob_state=prob_state)
    env_state = env_state.replace(env_map=dummy_env_map)
    # path_coords = get_path_coords(env_state.flood_count, max_path_len=get_max_path_length(env.map_shape))
    frame = env.render(env_state)
    # Save the frame
    imageio.imwrite(f"{exp_dir}/ep-{ep_i}_step-{env_state.step_idx}.png", frame, overwrite=True)
    breakpoint()

if __name__ == '__main__':
    main_test_dum_map()