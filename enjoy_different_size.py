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
def main_enjoy(config: EnjoyConfig):
    config = init_config(config)

    exp_dir = config.exp_dir
    if not config.random_agent:
        print(f'Loading checkpoint from {exp_dir}')
        checkpoint_manager, restored_ckpt = init_checkpointer(config)
        network_params = restored_ckpt['runner_state'].train_state.params
    elif not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    env: PCGRLEnv
    if config.eval_map_width is not None:
        config.map_width = config.eval_map_width
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env.prob.init_graphics()
    network = init_network(env, env_params, config)

    rng = jax.random.PRNGKey(42)
    rng_reset = jax.random.split(rng, config.n_eps)

    # Can manually define frozen tiles here, e.g. to set an OOD task
    frz_map = jnp.zeros(env.map_shape, dtype=bool)
    # frz_map = frz_map.at[7, 3:-3].set(1)
    queued_state = gen_dummy_queued_state(env)
    queued_state = env.queue_frz_map(queued_state, frz_map)

    # obs, env_state = env.reset(rng, env_params)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None, None))(
        rng_reset, env_params, queued_state
    )

    def step_env(carry, _):
        rng, obs, env_state = carry
        rng, rng_act = jax.random.split(rng)
        if config.random_agent:
            action = env.action_space(env_params).sample(rng_act)
        else:
            # obs = jax.tree_map(lambda x: x[None, ...], obs)
            action = network.apply(network_params, obs)[
                0].sample(seed=rng_act)
        rng_step = jax.random.split(rng, config.n_eps)
        # obs, env_state, reward, done, info = env.step(
        #     rng_step, env_state, action[..., 0], env_params
        # )
        obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            rng_step, env_state, action, env_params

        )
        frames = jax.vmap(env.render, in_axes=(0))(env_state)
        # frame = env.render(env_state)
        rng = jax.random.split(rng)[0]
        # Can't concretize these values inside jitted function (?)
        # So we add the stats on cpu later (below)
        # frame = render_stats(env, env_state, frame)
        return (rng, obs, env_state), (env_state, reward, done, info, frames)

    step_env = jax.jit(step_env)
    print('Scanning episode steps:')
    _, (states, rewards, dones, infos, frames) = jax.lax.scan(
        step_env, (rng, obs, env_state), None,
        length=1*env.max_steps)

    # frames = frames.reshape((config.n_eps*env.max_steps, *frames.shape[2:]))

    # assert len(frames) == config.n_eps * env.max_steps, \
    #     "Not enough frames collected"
    assert frames.shape[1] == config.n_eps and frames.shape[0] == env.max_steps, \
        "`frames` has wrong shape"


    # Save gifs.
    print('Adding stats to frames:')
    for ep_is in range(config.n_eps):
        # ep_frames = frames[ep_is*env.max_steps:(ep_is+1)*env.max_steps]

        new_ep_frames = []
        for i in range(env.max_steps):
            frame = frames[i, ep_is]
            
            state_i = jax.tree_util.tree_map(lambda x: x[i, ep_is], states)
            frame = render_stats(env, state_i, frame)
            new_ep_frames.append(frame)

            # Save frame as png
            png_name = f"{exp_dir}/frame_ep-{ep_is}_step-{i}" + \
                f"{('_randAgent' if config.random_agent else '')}.png"
            # imageio.v3.imwrite(png_name, frame)
            # imageio.imwrite(png_name, frame)
            new_ep_frames.append(frame)

        ep_frames = new_ep_frames

        frame_shapes = [frame.shape for frame in ep_frames]
        max_frame_w, max_frame_h = max(frame_shapes, key=lambda x: x[0])[0], \
            max(frame_shapes, key=lambda x: x[1])[1]
        # Pad frames to be same size
        new_ep_frames = []
        for frame in ep_frames:
            frame = np.pad(frame, ((0, max_frame_w - frame.shape[0]),
                                      (0, max_frame_h - frame.shape[1]),
                                      (0, 0)), constant_values=0)
            frame[:, :, 3] = 255
            new_ep_frames.append(frame)
        ep_frames = new_ep_frames

        # cum_rewards = jnp.cumsum(jnp.array(
        #   rewards[ep_is*env.rep.max_steps:(ep_is+1)*env.rep.max_steps]))
        gif_name = f"{exp_dir}/anim_ep-{ep_is}" + \
            f"{('_randAgent' if config.random_agent else '')}" + \
            f"_evalMapWidth{config.eval_map_width}" + ".gif"
        imageio.v3.imwrite(
            gif_name,
            ep_frames,
            # Not sure why but the frames are too slow otherwise (compared to 
            # when captured in `train.py`). Are we saving extra frames?
            duration=config.gif_frame_duration / 2
        )


if __name__ == '__main__':
    main_enjoy()
