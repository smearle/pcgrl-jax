"""Profile environment speed while taking random actions."""
import hydra
import jax
from timeit import default_timer as timer

from config import ProfileEnvConfig
from train import Dense
from utils import get_exp_dir, gymnax_pcgrl_make


@hydra.main(version_base=None, config_path='./', config_name='profile')
def enjoy(config: ProfileEnvConfig):

    exp_dir = get_exp_dir(config)

    env, env_params = gymnax_pcgrl_make(config.env_name)
    network = Dense(
        env.action_space(env_params).n, activation=config.activation
    )

    # state_seq, reward_seq = [], []
    rng = jax.random.PRNGKey(42)
    n_steps = 0

    start = timer()
    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    def _env_step(carry, unused):
        env_state, rng = carry
        rng, _rng = jax.random.split(rng)
        rng_act = jax.random.split(_rng, config.num_envs)
        action = jax.vmap(env.action_space(env_params).sample, in_axes=(0))(rng_act,)

        # STEP ENV
        rng_step = jax.random.split(_rng, config.num_envs)
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, action, env_params)
        carry = (env_state, rng)
        return carry, None

    carry = (env_state, rng)
    carry, _ = jax.lax.scan(
        _env_step, carry, None, config.N_PROFILE_STEPS
    )

    n_env_steps = config.N_PROFILE_STEPS * config.num_envs

    end = timer()
    print(f'Finished {n_env_steps} steps in {end - start} seconds.')
    print(f'Average steps per second: {n_env_steps / (end - start)}')

if __name__ == '__main__':
    enjoy() 