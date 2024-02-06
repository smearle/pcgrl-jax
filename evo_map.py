# Generate a random initial map and evaluate its fitness

import hydra
import jax
from envs.probs.problem import get_reward
from utils import gymnax_pcgrl_make

@hydra.main(config_path="./", config_name="evo_map_pcgrl")
def main(config):
    rng = jax.random.PRNGKey(config.seed)
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    rand_map = env.prob.gen_init_map(rng)

    # Evaluate the map
    env_stats = env.prob.get_curr_stats(rand_map)
    new_state = env.prob.get_curr_stats(env_map=rand_map)
    breakpoint()

if __name__ == '__main__':
    main()