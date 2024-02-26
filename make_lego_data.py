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
"""

def render_mpds(blocks, i, states=None):


            for ep_is in range(config.n_render_eps):
                ep_blocks = blocks[ep_is*env.max_steps:(ep_is+1)*env.max_steps]
                ep_avg_height = states.prob_state.stats[-1, ep_is, 0]
                ep_footprint = states.prob_state.stats[-1, ep_is, 1]
                ep_rotation_0 = states.rep_state.rotation[0, ep_is]
                ep_rotation_1 = states.rep_state.rotation[1, ep_is]
                ep_rotation_end = states.rep_state.rotation[-2, ep_is]
                ep_rotations = states.rep_state.rotation[:,ep_is]
                ep_curr_blocks = states.rep_state.curr_block[:,ep_is]
                actions = states.rep_state.last_action[:,ep_is]


                savedir = f"{config.exp_dir}/mpds/update-{i}_ep{ep_is}_ht{ep_avg_height:.2f}_fp{ep_footprint:.2f}_ro{ep_rotation_0}{ep_rotation_1}{ep_rotation_end}/"
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                
                for num in range(ep_blocks.shape[0]):
                    curr_blocks = ep_blocks[num,:,:]
                    rotation = ep_rotations[num]
                    curr_block = ep_curr_blocks[num]
                    action = actions[num]

                    savename = os.path.join(savedir, f"{num}_r{rotation}_a{action}.mpd")
                    
                    f = open(savename, "a")
                    f.write("0/n")
                    f.write("0 Name: New Model.ldr\n")
                    f.write("0 Author:\n")
                    f.write("\n")

                    for x in range(config.map_width):
                        for z in range(config.map_width):
                            lego_block_name = "3005"
                            block_color = "2 "
                            if curr_block == num:
                                block_color = "7 "
                            
                            y_offset = -3#-24

                            x_lego = x * 20  + config.map_width - 1
                            y_lego =  0#(1)*(LegoDimsDict[lego_block_name][1])
                            z_lego = z * 20 + config.map_width - 1

                            #print(block.x, block.y, block.z)
                            
                            f.write("1 ")
                            f.write(block_color)
                            f.write(str(x_lego) + ' ' + str(y_lego) + ' ' + str(z_lego) + ' ')
                            f.write("1 0 0 0 1 0 0 0 1 ")
                            f.write(lego_block_name + ".dat")
                            f.write("\n")
                    
                    y_offset = -24
                    for b in range(curr_blocks.shape[0]):
                        lego_block_name = "3005"
                        block_color = "7 "
                        x_lego = curr_blocks[b, 0] * 20  + config.map_width - 1
                        y_lego = curr_blocks[b, 1] * (-24) + y_offset
                        z_lego = curr_blocks[b, 2] * 20 + config.map_width - 1

                        f.write("1 ")
                        f.write(block_color)
                        f.write(str(x_lego) + ' ' + str(y_lego) + ' ' + str(z_lego) + ' ')
                        f.write("1 0 0 0 1 0 0 0 1 ")
                        f.write(lego_block_name + ".dat")
                        f.write("\n")
                    f.close()
"""
moves = np.array([
            (0,0),
            (0,1),
            (0,-1),
            (1,0),
           # (1,1),
            #(1,-1),
            (-1,0),
            #(-1,1),
            #(-1,-1)   
        ])

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

    indices = np.where(np.all(moves == (x_action, z_action), axis=1))
    return jnp.array([[indices]])

@hydra.main(version_base=None, config_path='./', config_name='lego_pcgrl')
def main(config: TrainConfig):
        
    dataset_size = 100

    rng = jax.random.PRNGKey(42)
    rng, subkey = jax.random.split(rng)

    savedir = os.path.join(os.getcwd(), "saves", "lego_data")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    
    
    obs, env_state = env.reset(subkey)
    done = False
    observations = []
    actions = []

    observations_arr = np.empty((dataset_size,) + obs.map_obs.shape)
    #actions_arr = np.empty((dataset_size,))

    for i in tqdm(range(dataset_size)):
        rng, subkey = jax.random.split(rng)
        if done:
            flatmap = jnp.count_nonzero(env_state.env_map, 1)
            obs, env_state = env.reset_env(subkey, env_params, None)
            rng, subkey = jax.random.split(rng)
            
            
        
        action = get_action(env_state)
        print(action)

        actions.append(action[0][0][0])
        observations.append(obs.map_obs)

        observations_arr[i] = obs.map_obs
        #actions_arr[i] = action[0][0]
        #obs_action_pairs.append((obs, action))
        obs, env_state, reward, done, info= env.step(subkey, env_state, action, env_params)
        #print(i, reward)

    obs_fn = os.path.join(savedir, "observations.npz")
    np.savez(obs_fn, *observations)

    act_fn = os.path.join(savedir, "actions.npz")
    np.savez(act_fn, *actions)
    

    obs_action_fn = os.path.join(savedir, "obs_action_pairs.npz")
    #np.savez_compressed(obs_action_fn, actions = actions_arr, observations = observations_arr)
    
    tmp = np.load(obs_fn)
    tmp2 = np.load(act_fn)
    #tmp3 = np.load(obs_action_fn)
    test = 0




if __name__ == "__main__":
    main()