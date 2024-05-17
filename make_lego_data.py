import os
import numpy as np
import jax
import hydra
import jax.numpy as jnp
from tqdm import tqdm


from envs.probs.lego import LegoProblemState
from envs.lego_env import LegoEnvState, LegoEnvParams
from utils import (get_ckpt_dir, get_exp_dir, get_network, gymnax_pcgrl_make,
                   init_config)
from config import Config, TrainConfig

moves = jnp.array([
            (0,0), #no move, goes to top
            (0,1),
            (0,-1),
            (1,0),
            (-1,0),
            (0,0) # no move, does not go to top  
        ])


def compute_returns(rewards, discount_factor):
    """
    Compute discounted returns.
    Args:
        rewards (jnp.ndarray): An array of shape (batch_size, timesteps) representing
            rewards collected after each action.
        discount_factor (float): Discount factor (gamma), should be in range [0, 1].
    Returns:
        jnp.ndarray: Array of shape (batch_size, timesteps) representing the
            discounted returns for each time step.
    """
    length = rewards.shape[0]
    returns = jnp.zeros_like(rewards)
    next_return = 0  # next_return is the accumulated reward from the next timestep onwards

    for t in reversed(range(length)):
        next_return = rewards[t] + discount_factor * next_return
        returns = returns.at[t].set(next_return)

    return returns

def get_action(env_state):
    blocks = env_state.rep_state.blocks
    curr_block = env_state.rep_state.curr_block
    map_shape = env_state.env_map.shape

    ctr_x, ctr_z = (map_shape[0]-1)//2, (map_shape[2]-1)//2
    curr_x, curr_z = blocks[curr_block][0], blocks[curr_block][2]
    x_dir = ctr_x - curr_x
    z_dir = ctr_z - curr_z

    x_action = -1 if x_dir < 0 else 0 if x_dir == 0 else 1
    z_action = -1 if z_dir < 0 else 0 if z_dir == 0 else 1
    z_action = 0 if x_action != 0 else z_action

    indices = np.where(np.all(moves == (x_action, z_action), axis=1))
    return jnp.array([indices])#.reshape((1,1,1))

@hydra.main(version_base=None, config_path='./', config_name='lego_pcgrl')
def main(config: TrainConfig):
        
    dataset_size = 1000

    rng = jax.random.PRNGKey(42)
    rng, subkey = jax.random.split(rng)

    savedir = os.path.join(os.getcwd(), "saves", "lego_data")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    
    obs, env_state = env.reset(subkey)
    done = False

    observations_arr = np.empty((dataset_size,) + obs.map_obs.shape)
    actions_arr = np.empty((dataset_size,))
    rewards_arr = np.empty((dataset_size,))

    for i in tqdm(range(dataset_size)):
        rng, subkey = jax.random.split(rng)
        if done:
            flatmap = jnp.count_nonzero(env_state.env_map, 1)
            obs, env_state = env.reset_env(subkey, env_params, None)
            rng, subkey = jax.random.split(rng)
            
        action = get_action(env_state)
        
        observations_arr[i] = obs.map_obs
        actions_arr[i] = action[0][0]
        rewards_arr[i] = env_state.reward
        #obs_action_pairs.append((obs, action))
        obs, env_state, reward, done, info= env.step(subkey, env_state, action, env_params)
        #print(i, reward)

   

    returns = compute_returns(rewards_arr, 0.5)
    obs_action_fn = os.path.join(savedir, "obs_action_pairs.npz")
    np.savez_compressed(obs_action_fn, returns = returns, actions = actions_arr, observations = observations_arr, rewards = rewards_arr)
    np.savez_compressed(obs_action_fn, actions = actions_arr, observations = observations_arr)
   
    
    tmp3 = np.load(obs_action_fn)
    t = tmp3["actions"]
    test = 0




if __name__ == "__main__":
    main()