import imageio
import jax
import jax.numpy as jnp

import envs.pcgrl_env
import hydra

from envs.pcgrl_env import gen_dummy_queued_state
from utils import (get_ckpt_dir, get_exp_dir, init_network, gymnax_pcgrl_make,
        init_config)

# Debugging flag for development DO NOT SHIP WITH THIS ON
DEBUG = False 
if DEBUG:
    import sys
    jnp.set_printoptions(threshold=sys.maxsize)
    print("-----------***------------")
    print("Code running in DEBUG MODE")
    print("  !THIS *DISABLES* JIT!") 
    print("----------*****-----------")

jax.config.update('jax_disable_jit', DEBUG)

@hydra.main(version_base=None, config_path='./conf', config_name='train_pcgrl')
def main(config):
    config = init_config(config)
    env, env_params = gymnax_pcgrl_make(config.env_name, config)
    env.init_graphics()
    print(f"Created env with {env_params.n_agents} agents: {env_params} ")

    dummy_queued_state = gen_dummy_queued_state(env)
    obs, state = env.reset_env(jax.random.key(0), env_params, dummy_queued_state)

    vmap_sample_action = jax.vmap(env.sample_action)

    frames = []
    rng = jax.random.key(94)
    for _ in range(1000):
        rng, this_rng, that_rng = jax.random.split(rng, 3)
        sample_action_keys = jax.random.split(that_rng, env_params.n_agents)
        actions = vmap_sample_action(sample_action_keys)
        obs, state, reward, done, info = env.step(this_rng, state, actions, env_params)
        frame = env.render(state)
        frames.append(frame)

    imageio.v3.imwrite('test11.gif', frames)
        
    print("Ran Successfully!")

if __name__ == '__main__':
    main()
