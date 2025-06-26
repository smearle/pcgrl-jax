import os

import hydra
import imageio
import jax

from conf.config import EnjoyConfig
from train import init_checkpointer
from utils import get_exp_dir, init_network, gymnax_pcgrl_make, init_config


@hydra.main(version_base="1.3", config_path='./', config_name='enjoy_pcgrl')
def main_enjoy(config: EnjoyConfig):
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

    obs, env_state = env.reset(rng, env_params)

    def step_env(carry, _):
        rng, obs, env_state = carry
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if config.random_agent:
            action = env.action_space(env_params).sample(rng_act)
        else:
            action = network.apply(network_params, obs[None])[
                0].sample(seed=rng_act)
        obs, env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        frames = info['frames']
        return (rng, obs, env_state), (env_state, reward, done, info, frames)

    # step_env = jax.jit(step_env)
    print('Scanning episode steps:')
    # _, (states, rewards, dones, infos, frames) = jax.lax.scan(
    #     step_env, (rng, obs, env_state), None,
    #     length=config.n_eps*env.rep.max_steps)
    for ep_i in range(config.n_eps):
        ep_frames = []
        for _ in range(env.max_steps):
            (rng, obs, env_state), (env_state, reward, done, info, frames) = step_env((rng, obs, env_state), None)
            print(f"step_env: {reward}")
            ep_frames.append(frames)

        ep_frames = [env.render]

        # Save gifs.
        # cum_rewards = jnp.cumsum(jnp.array(
        #   rewards[ep_is*env.rep.max_steps:(ep_is+1)*env.rep.max_steps]))
        gif_name = f"{exp_dir}/anim_ep-{ep_i}" + \
            f"{('_randAgent' if config.random_agent else '')}.gif"
        imageio.v3.imwrite(
            gif_name,
            ep_frames,
            duration=config.duration
        )


if __name__ == '__main__':
    main_enjoy()
