import os

import hydra
import imageio
import jax

from config import EnjoyConfig
from envs.pcgrl_env import render_stats
from train import init_checkpointer
from utils import get_exp_dir, get_network, gymnax_pcgrl_make, init_config


@hydra.main(version_base=None, config_path='./', config_name='enjoy_pcgrl')
def main_enjoy(config: EnjoyConfig):
    config = init_config(config)

    exp_dir = get_exp_dir(config)
    if not config.random_agent:
        checkpoint_manager, restored_ckpt = init_checkpointer(config)
        network_params = restored_ckpt['runner_state'].train_state.params
    elif not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env.prob.init_graphics()
    network = get_network(env, env_params, config)

    rng = jax.random.PRNGKey(42)

    obs, env_state = env.reset(rng, env_params)

    def step_env(carry, _):
        rng, obs, env_state = carry
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if config.random_agent:
            action = env.action_space(env_params).sample(rng_act)
        else:
            obs = jax.tree_map(lambda x: x[None, ...], obs)
            action = network.apply(network_params, obs)[
                0].sample(seed=rng_act)
        obs, env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        rng = jax.random.split(rng, 1)[0]
        frame = env.render(env_state)
        # Can't concretize these values inside jitted function (?)
        # So we add the stats on cpu later (below)
        # frame = render_stats(env, env_state, frame)
        return (rng, obs, env_state), (env_state, reward, done, info, frame)

    step_env = jax.jit(step_env)
    print('Scanning episode steps:')
    _, (states, rewards, dones, infos, frames) = jax.lax.scan(
        step_env, (rng, obs, env_state), None,
        length=config.n_eps*env.max_steps)


    assert len(frames) == config.n_eps * env.max_steps, \
        "Not enough frames collected"

    # Save gifs.
    for ep_is in range(config.n_eps):
        ep_frames = frames[ep_is*env.max_steps:(ep_is+1)*env.max_steps]

        print('Adding stats to frames:')
        new_ep_frames = []
        for i, frame in enumerate(ep_frames):
            state_i = jax.tree_util.tree_map(lambda x: x[ep_is*env.max_steps+i], states)
            frame = render_stats(env, state_i, frame)
            new_ep_frames.append(frame)

            # Save frame as png
            png_name = f"{exp_dir}/frame_ep-{ep_is}_step-{i}" + \
                f"{('_randAgent' if config.random_agent else '')}.png"
            # imageio.v3.imwrite(png_name, frame)
            # imageio.imwrite(png_name, frame)
            new_ep_frames.append(frame)

        ep_frames = new_ep_frames

        # cum_rewards = jnp.cumsum(jnp.array(
        #   rewards[ep_is*env.rep.max_steps:(ep_is+1)*env.rep.max_steps]))
        gif_name = f"{exp_dir}/anim_ep-{ep_is}" + \
            f"{('_randAgent' if config.random_agent else '')}.gif"
        imageio.v3.imwrite(
            gif_name,
            ep_frames,
            duration=config.gif_frame_duration
        )


if __name__ == '__main__':
    main_enjoy()
