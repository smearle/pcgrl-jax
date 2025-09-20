import os

import hydra
import imageio
import jax
import jax.numpy as jnp
import distrax

from conf.config import EnjoyConfig
from envs.pcgrl_env import PCGRLEnvState, render_stats
from train import init_checkpointer
from utils import get_exp_dir, init_network, gymnax_pcgrl_make, init_config


@hydra.main(version_base="1.3", config_path='./', config_name='enjoy_pcgrl')
def main_enjoy_cpu(config: EnjoyConfig):
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
    latent = jnp.zeros(obs.map_obs.shape[:-1] + (config.nca_latent_dim,), dtype=jnp.float32) if config.model == "nca" else jnp.zeros((1,), dtype=jnp.float32)
    use_mask = config.model == "nca" and config.representation == "nca" and config.nca_mask_keep_prob < 1.0

    def apply_latent_network(params, obs, latent_state, rng_mask):
        if use_mask:
            return network.apply(params, obs, latent_state, rngs={'nca_mask': rng_mask})
        return network.apply(params, obs, latent_state)
    frames = [env.render(env_state)]
    ep_i = 0

    while True:
        rng, rng_apply, rng_act, rng_step = jax.random.split(rng, 4)
        if config.random_agent:
            action = env.action_space(env_params).sample(rng_act)
        else:
            # Apply tree map to obs so that all leaves have a batch of size 1
            obs = jax.tree_map(lambda x: x[None, ...], obs) 
            if config.model == "nca":
                logits, _, latent = apply_latent_network(network_params, obs, latent, rng_apply)
                pi = distrax.Categorical(logits=logits)
                action = pi.sample(seed=rng_act)[0]
            else:
                action = network.apply(network_params, obs)[
                    0].sample(seed=rng_act)
        obs, env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        if config.model == "nca":
            reset_mask = jnp.reshape(jnp.asarray(done), (1,) + (1,) * (latent.ndim - 1))
            latent = jnp.where(reset_mask, jnp.zeros_like(latent), latent)
        frame = env.render(env_state)
        frame = render_stats(env, env_state, frame)

        # Save frame as image
        imageio.imwrite(f"{exp_dir}/ep-{ep_i}_step-{env_state.step_idx}.png", frame)

        frames.append(frame)

        # If done, save gif
        if done:
            imageio.mimsave(f"{exp_dir}/ep-{ep_i}.gif", frames)
            frames = []

        # Print control targets, weights, stats, and ctrl obs
        print(f'ctrl_metrics: {env.prob.ctrl_metrics}')
        print(f'ctrl_trgs: {env_state.prob_state.ctrl_trgs}')
        print(f'stat_weights: {env.prob.stat_weights}')
        print(f'stats: {env_state.prob_state.stats}')
        print(f'ctrl_obs: {obs.prob_obs}')
        print()

        # print(env_state.prob.ctrl_trgs)
        # print(env_state.env_map)
        # print(reward)
        # frame = env.render(env_state)
        # return (rng, obs, env_state), (env_state, reward, done, info, frame)


if __name__ == '__main__':
    main_enjoy_cpu()
