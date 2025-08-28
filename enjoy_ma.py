import copy
import os

import hydra
import imageio
import jax
from jax import numpy as jnp
import numpy as np
import orbax
from orbax import checkpoint as ocp

from conf.config import EnjoyConfig, EnjoyMultiAgentConfig
from envs.pcgrl_env import PCGRLEnv, render_stats, gen_dummy_queued_state
from envs.probs.problem import get_loss
from eval import get_eval_name, init_config_for_eval
from utils_ma import MALogWrapper, MultiAgentWrapper, batchify, init_run, ma_init_config, make_sim_render_episode, render_callback, restore_run
from marl.model import ScannedRNN
from purejaxrl.experimental.s5.wrappers import LossLogWrapper
from utils import get_exp_dir, init_network, gymnax_pcgrl_make, init_config


@hydra.main(version_base="1.3", config_path='./conf', config_name='enjoy_ma_pcgrl')
def main_enjoy_ma(enjoy_config: EnjoyMultiAgentConfig):
    # enjoy_config = init_config(enjoy_config)
    ma_init_config(enjoy_config)
    rng = jax.random.PRNGKey(enjoy_config.eval_seed)

    exp_dir = enjoy_config._exp_dir
    if not enjoy_config.random_agent:
        print(f'Loading checkpoint from {exp_dir}')
        options = ocp.CheckpointManagerOptions(
            max_to_keep=2, create=True)
        checkpoint_manager = ocp.CheckpointManager(
            enjoy_config._ckpt_dir,
            options=options)
        latest_update_step = checkpoint_manager.latest_step()

        runner_state, actor_network, env, latest_update_step = \
            init_run(enjoy_config, checkpoint_manager, latest_update_step, rng)

        assert latest_update_step is not None
        runner_state, wandb_run_id = restore_run(enjoy_config, runner_state, checkpoint_manager, latest_update_step)
        network_params = runner_state.train_states[0].params
    else:
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        steps_prev_complete = 0
    
    env: PCGRLEnv

    # Preserve config as it was during training, for future reference (i.e. naming output of enjoy/eval)
    train_config = copy.deepcopy(enjoy_config)

    enjoy_config = init_config_for_eval(enjoy_config)
    env, env_params = gymnax_pcgrl_make(enjoy_config.env_name, config=enjoy_config)

    # Wrap environment with JAXMARL wrapper
    env = MultiAgentWrapper(env, env_params)

    # Wrap environment with LogWrapper
    env = MALogWrapper(env)

    env.prob.init_graphics()
    # network = init_network(env, env_params, enjoy_config)

    rng = jax.random.PRNGKey(enjoy_config.eval_seed)
    rng_reset = jax.random.split(rng, enjoy_config.n_enjoy_envs)

    jit_sim_render_episode = make_sim_render_episode(enjoy_config, actor_network, env)
    num_render_actors = 1 * env.n_agents
    ac_init_hstate_render = ScannedRNN.initialize_carry(num_render_actors, enjoy_config.hidden_dims[0])
    frames = jit_sim_render_episode(runner_state.train_states[0].params, ac_init_hstate_render)
    t = latest_update_step
    enjoy_vid_dir = os.path.join(exp_dir, 'enjoy')
    os.makedirs(enjoy_vid_dir, exist_ok=True)
    imageio.mimsave(os.path.join(enjoy_vid_dir, f"enjoy_{t}.gif"), np.array(frames), fps=20, loop=0)
    print(f"Saved enjoy video to {os.path.join(enjoy_vid_dir, f'enjoy_{t}.gif')}")

if __name__ == '__main__':
    main_enjoy_ma()
