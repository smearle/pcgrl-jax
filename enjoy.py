import os

import hydra
import imageio
import jax

from config import EnjoyConfig
from train import init_checkpointer
from utils import get_exp_dir, get_network, gymnax_pcgrl_make, init_config


N_EPS = 10


@hydra.main(version_base=None, config_path='./', config_name='enjoy')
def enjoy(config: EnjoyConfig):
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

    def step_env(carry, action):
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
        frame = env.render(env_state)
        return (rng, obs, env_state), (env_state, reward, done, info, frame)

    step_env = jax.jit(step_env)
    print('Scanning episode steps:')
    _, (states, rewards, dones, infos, frames) = jax.lax.scan(
        step_env, (rng, obs, env_state), None, length=N_EPS*env.rep.max_steps)

    assert len(frames) == N_EPS * env.rep.max_steps, "Not enough frames" + \
                                                     "collected"

    # Save gifs.
    for ep_is in range(N_EPS):
        # cum_rewards = jnp.cumsum(jnp.array(
        #   rewards[ep_is*env.rep.max_steps:(ep_is+1)*env.rep.max_steps]))
        gif_name = f"{exp_dir}/anim_ep-{ep_is}" + \
            f"{('_randAgent' if config.random_agent else '')}.gif"
        # imageio.mimsave(
        #     gif_name,
        #     frames[ep_is*env.rep.max_steps:(ep_is+1)*env.rep.max_steps],
        #     duration=1/30)
        imageio.v3.imwrite(
            gif_name,
            frames[ep_is*env.rep.max_steps:(ep_is+1)*env.rep.max_steps]
        )


if __name__ == '__main__':
    enjoy()
