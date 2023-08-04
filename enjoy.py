import os
import gymnax
# from gymnax.visualize import Visualizer
import hydra
import imageio
import jax
from jax import numpy as jnp
import orbax
from orbax import checkpoint
from config import Config, EnjoyConfig
import ray
from train import Dense, init_checkpointer

from utils import get_ckpt_dir, get_exp_dir, get_network, gymnax_pcgrl_make, init_config


N_EPS = 10


@hydra.main(version_base=None, config_path='./', config_name='enjoy')
def enjoy(config: EnjoyConfig):
    config = init_config(config)
    ckpt_dir = get_ckpt_dir(config)

    exp_dir = get_exp_dir(config)
    if not config.random_agent:
        checkpoint_manager, restored_ckpt = init_checkpointer(config)
        network_params = restored_ckpt['runner_state'].train_state.params
    elif not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env.prob.init_graphics()
    network = get_network(env, env_params, config)

    if config.multiproc:
        # Initialize Ray
        ray.init()

        # Create a Ray actor for the environment (if necessary)
        # env = ray.remote(env)

        @ray.remote
        def render_frame(state, params):
            # Perform the rendering operation here
            return env.render(state)

    rng = jax.random.PRNGKey(42)

    obs, env_state = env.reset(rng, env_params)

    def step_env(carry, action):
        rng, obs, env_state = carry
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if config.random_agent:
            action = env.action_space(env_params).sample(rng_act)
        else:
            action = network.apply(network_params, obs[None])[0].sample(seed=rng_act)
        obs, env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        frame = env.render(env_state)
        return (rng, obs, env_state), (env_state, reward, done, info, frame)

    step_env = jax.jit(step_env)
    print('Scanning episode steps:')
    _, (states, rewards, dones, infos, frames) = jax.lax.scan(step_env, (rng, obs, env_state), None, length=N_EPS*env.rep.max_steps)

    for ep_is in range(N_EPS):
        cum_rewards = jnp.cumsum(jnp.array(rewards[ep_is*env.rep.max_steps:(ep_is+1)*env.rep.max_steps]))
        gif_name = f"{exp_dir}/anim_ep-{ep_is}{('_randAgent' if config.random_agent else '')}.gif"
        imageio.mimsave(gif_name, frames[ep_is*env.rep.max_steps:(ep_is+1)*env.rep.max_steps], duration=40)

    # s_i = 0
    # for ep_i in range(N_EPS):

    #     cum_rewards = jnp.cumsum(jnp.array(rewards))
    #     gif_name = f"{exp_dir}/anim_ep-{ep_i}{('_randAgent' if config.random_agent else '')}.gif"
    #     # vis = Visualizer(env, env_params, state_seq, cum_rewards)
    #     # vis.animate(gif_name)

    #     if not config.multiproc:
    #         frames = [env.render(state) for state in states]

    #     else:
    #         # Create a list to store the references to the remote tasks
    #         frame_refs = []

    #         # Iterate over the state sequence and submit rendering tasks
    #         for state in states:
    #             frame_refs.append(render_frame.remote(state, env_params))

    #         # Get the rendered frames from the completed tasks
    #         frames = ray.get(frame_refs)

    #     # Save them as frames into a gif
    #     imageio.mimsave(gif_name, frames, duration=40)

    
    # Shutdown Ray
    ray.shutdown()




if __name__ == '__main__':
    enjoy() 