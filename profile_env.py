"""Profile environment speed while taking random actions."""
import hydra
import jax
import json
import pandas as pd
from timeit import default_timer as timer

from conf.config import ProfileEnvConfig
from utils import get_exp_dir, gymnax_pcgrl_make, init_config


problems = [
    'binary',
    'maze',
    'dungeon',
]

# n_envss= [1, 10, 50, 100, 200, 400]
n_envss= [1,2]


@hydra.main(version_base=None, config_path='./', config_name='profile_pcgrl')
def profile(config: ProfileEnvConfig):
    if config.reevaluate:
        config = init_config(config)
        exp_dir = config.exp_dir

        env, env_params = gymnax_pcgrl_make(config.env_name, config)
        # network = Dense(
        #     env.action_space(env_params).n, activation=config.activation
        # )

        # state_seq, reward_seq = [], []
        rng = jax.random.PRNGKey(42)
        n_steps = 0

        problem_n_envs_to_fps = {}
        
        for problem in problems:
            config.problem = problem
            problem_n_envs_to_fps[problem] = {}
            for n_envs in n_envss:
                config.n_envs= n_envs
                # INIT ENV
                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config.n_envs)
                obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

                def _env_step(carry, unused):
                    env_state, rng = carry
                    rng, _rng = jax.random.split(rng)
                    rng_act = jax.random.split(_rng, config.n_envs)
                    action = jax.vmap(env.action_space(env_params).sample, in_axes=(0))(rng_act,)
                    action = action[..., None, None, None]

                    # STEP ENV
                    rng_step = jax.random.split(_rng, config.n_envs)
                    obsv, env_state, reward, done, info = jax.vmap(
                        env.step, in_axes=(0, 0, 0, None)
                    )(rng_step, env_state, action, env_params)
                    carry = (env_state, rng)
                    return carry, None

                _env_step_jitted = jax.jit(_env_step)
                start = timer()
                carry = (env_state, rng)
                carry, _ = jax.lax.scan(
                    _env_step_jitted, carry, None, config.N_PROFILE_STEPS
                )

                n_env_steps = config.N_PROFILE_STEPS * config.n_envs

                end = timer()
                print(f'Finished {n_env_steps} steps in {end - start} seconds.')
                fps = n_env_steps / (end - start)
                print(f'Average steps per second: {fps}')

                problem_n_envs_to_fps[problem][n_envs] = fps

        # Save as json
        with open(f'n_envs_to_fps.json', 'w') as f:
            json.dump(problem_n_envs_to_fps, f)

    else:
        # Load from json
        with open(f'n_envs_to_fps.json', 'r') as f:
            problem_n_envs_to_fps = json.load(f)

    # Turn into a dataframe, where rows are problems and columns are different n_envs
    fps_df = pd.DataFrame(problem_n_envs_to_fps)
    print(fps_df)
    # Save as markdown
    fps_df.to_markdown(f'n_envs_to_fps.md')
    # latex
    fps_df.to_latex(f'n_envs_to_fps.tex')

if __name__ == '__main__':
    profile() 
