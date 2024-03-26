import os
import shutil
from timeit import default_timer as timer
from typing import NamedTuple
from matplotlib import pyplot as plt
import hydra
import jax
import jax.numpy as jnp
from flax import struct
import imageio
import optax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
from flax.training.common_utils import onehot
import orbax 
from tensorboardX import SummaryWriter

from config import Config, TrainConfig
from envs.pcgrl_env import PCGRLObs, QueuedState, gen_static_tiles, render_stats
from purejaxrl.experimental.s5.wrappers import LogWrapper
from utils import (get_ckpt_dir, get_exp_dir, get_network, gymnax_pcgrl_make, init_config)
from envs.probs.lego import LegoProblemState
from envs.lego_env import LegoEnvState

class RunnerState(struct.PyTreeNode):
    train_state: TrainState
    env_state: jnp.ndarray
    last_obs: jnp.ndarray
    # rng_act: jnp.ndarray
#   ep_returns: jnp.ndarray
    rng: jnp.ndarray
    update_i: int


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    expert_action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    # rng_act: jnp.ndarray

def get_expert_action(env_state):
    #env_state = log_env_state.env_state
    moves = jnp.array([
            (0,0),
            (0,1),
            (0,-1),
            (1,0),
            (-1,0),
        ])
    
    blocks = env_state.rep_state.blocks
    curr_block_inds = env_state.rep_state.curr_block
    map_shape = env_state.env_map[0].shape

    n_envs = blocks.shape[0]

    center_x = (map_shape[0]-1)//2
    center_z = (map_shape[2]-1)//2

    curr_blocks = jnp.array([blocks[i, curr_block_inds[i], :] for i in range(blocks.shape[0])])
    curr_x = curr_blocks[:,0]
    curr_z = curr_blocks[:,2]
    
    x_dir = center_x - curr_x
    z_dir = center_z - curr_z

    x_moves = jnp.clip(x_dir, a_min = -1, a_max = 1).reshape(x_dir.shape[0], 1)
    z_moves = jnp.clip(z_dir, a_min = -1, a_max = 1).reshape(x_dir.shape[0], 1)
    z_moves = jnp.where(x_moves==0, z_moves, 0)
    
    combined_actions = jnp.concatenate((x_moves, z_moves), axis=1)

    def find_indices(array1, array2):
        # Step 1: Compare each row of array1 with each row of array2
        comparison = array1[:, None, :] == array2[None, :, :]  # Shape becomes (4, 5, 2)

        # Step 2: Check where all elements along the last axis match
        matches = jnp.all(comparison, axis=2)  # Shape becomes (4, 5)

        # Step 3: Find the index in the second array for each row of the first array
        indices = jnp.argmax(matches, axis=1)  # Shape becomes (4,)

        return indices

    actions = find_indices(combined_actions, moves)
    
    jax.debug.print("goal: {goalx} {goaly}", goalx = center_x, goaly=center_z)
    jax.debug.print("current x pos: {x}.\n current z pos: {z}.\n xmoves: {xmoves}.\n zmoves: {zmoves}.\n combined: {combined_actions}.\n actions: {acts}", x = curr_x, z = curr_z, xmoves = x_moves, zmoves = z_moves, acts = actions, combined_actions = combined_actions)
    
    actions = actions.reshape(actions.shape[0], 1, 1, 1)
    return actions


def make_train(config: TrainConfig, restored_ckpt, checkpoint_manager):
    config.NUM_UPDATES = (
        config.total_timesteps // (config.max_steps_multiple*config.n_blocks )// config.n_envs
    )
    config.MINIBATCH_SIZE = (
        config.n_envs * config.n_blocks * config.max_steps_multiple // config.NUM_MINIBATCHES
    )
    env_r, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env = LogWrapper(env_r)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.NUM_MINIBATCHES * config.update_epochs))
            / config.NUM_UPDATES
        )
        return config["LR"] * frac
        

    def train(rng, config: TrainConfig):

        train_start_time = timer()

        # INIT NETWORK
        network = get_network(env, env_params, config)
        rng, _rng = jax.random.split(rng)
        init_x = env.gen_dummy_obs(env_params)
     
        network_params = network.init(_rng, init_x)
        print(network.subnet.tabulate(_rng, init_x.map_obs, init_x.flat_obs))

        # Print number of learnable parameters in the network
        if config.ANNEAL_LR:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(config.lr, eps=1e-5),
            )
       
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV FOR TRAIN
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.n_envs)
 
        dummy_queued_state = gen_dummy_queued_state(config, env, reset_rng)

        vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
        obsv, env_state = vmap_reset_fn(reset_rng, env_params, dummy_queued_state)
        
        # INIT ENV FOR RENDER
        rng_r, _rng_r = jax.random.split(rng)
        reset_rng_r = jax.random.split(_rng_r, config.n_render_eps)

        vmap_reset_fn = jax.vmap(env_r.reset, in_axes=(0, None, None))
        obsv_r, env_state_r = vmap_reset_fn(reset_rng_r, env_params, dummy_queued_state)  # Replace None with your env_params if any

        rng, _rng = jax.random.split(rng)

        steps_prev_complete = 0
        runner_state = RunnerState(
            train_state, env_state, obsv, rng,
            update_i=0)

        
        if restored_ckpt is not None:
            steps_prev_complete = restored_ckpt['steps_prev_complete']
            runner_state = restored_ckpt['runner_state']
            steps_remaining = config.total_timesteps - steps_prev_complete
            config.NUM_UPDATES = int(
                steps_remaining // config.num_steps // config.n_envs)

            # TODO: Overwrite certain config values

        
        def render_frames(frames, i, env_states=None):
            if i % config.render_freq != 0:
            # if jnp.all(frames == 0):
                return
            
            is_finished = env_states.done
           
        
            # Save gifs.
            for ep_is in range(config.n_render_eps):
                gif_name = f"{config.exp_dir}/update-{i}_ep-{ep_is}.gif"
                dones = is_finished[:,ep_is]
                done_ind = jnp.argmax(dones)
                done_ind = jax.lax.cond(done_ind == 0, lambda: env.max_steps-1, lambda: done_ind)

                ep_frames = frames[ep_is*env.max_steps:(ep_is+1)*env.max_steps]
                ep_frames = ep_frames[:done_ind]

                try:
                    imageio.v3.imwrite(
                        gif_name,
                        ep_frames,
                        duration=config.gif_frame_duration
                    )
                except jax.errors.TracerArrayConversionError:
                    print("Failed to save gif. Skipping...")
                    return
            print(f"Done rendering episode gifs at update {i}")

        def render_mpds(blocks, i, states):

            if i % config.render_freq != 0:
                return
            

            for ep_is in range(config.n_render_eps):
                dones = is_finished[:,ep_is]
                done_ind = jnp.argmax(dones)
                done_ind = jax.lax.cond(done_ind == 0, lambda: env.max_steps-1, lambda: done_ind)

                ep_blocks = blocks[ep_is*env.max_steps:(ep_is+1)*env.max_steps]

                ep_end_avg_height = states.prob_state.stats[done_ind-1, ep_is, 0]
                ep_footprint = states.prob_state.stats[done_ind-1, ep_is, 1]
                ep_cntr_dist = states.prob_state.stats[done_ind-1, ep_is, 3]
                #ep_rotations = states.rep_state.rotation[:done_ind+1,ep_is]
                ep_curr_blocks = states.rep_state.curr_block[:,ep_is]
                actions = states.rep_state.last_action[:,ep_is]

                savedir = f"{config.exp_dir}/mpds/update-{i}_ep{ep_is}_ht{ep_end_avg_height:.2f}_fp{ep_footprint:.2f}_ctrdist{ep_cntr_dist:.2f}/"
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                
                for num in range(ep_blocks.shape[0]):
                    curr_blocks = ep_blocks[num,:,:]
                    #rotation = ep_rotations[num]
                    curr_block = ep_curr_blocks[num]
                    action = actions[num]

                    savename = os.path.join(savedir, f"{num}_a{action}.mpd")
                    
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


            if i % config.render_freq != 0:
                return
            
            is_finished = states.done

            for ep_is in range(config.n_render_eps):
                dones = is_finished[:,ep_is]
                done_ind = jnp.argmax(dones)

                ep_blocks = blocks[ep_is*env.max_steps:(ep_is+1)*env.max_steps]

                ep_end_avg_height = states.prob_state.stats[done_ind-1, ep_is, 0]
                ep_footprint = states.prob_state.stats[done_ind-1, ep_is, 1]
                ep_cntr_dist = states.prob_state.stats[done_ind-1, ep_is, 3]
                #ep_rotations = states.rep_state.rotation[:done_ind+1,ep_is]
                ep_curr_blocks = states.rep_state.curr_block[:,ep_is]
                actions = states.rep_state.last_action[:,ep_is]

                savedir = f"{config.exp_dir}/mpds/update-{i}_ep{ep_is}_ht{ep_end_avg_height:.2f}_fp{ep_footprint:.2f}_ctrdist{ep_cntr_dist:.2f}/"
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                
                for num in range(ep_blocks.shape[0]):
                    curr_blocks = ep_blocks[num,:,:]
                    #rotation = ep_rotations[num]
                    curr_block = ep_curr_blocks[num]
                    action = actions[num]

                    savename = os.path.join(savedir, f"{num}_a{action}.mpd")
                    
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

        def render_episodes(network_params):
            _, (states, rewards, dones, infos, frames, blocks) = jax.lax.scan(
                step_env_render, (rng_r, obsv_r, env_state_r, network_params),
                None, 1*env.max_steps)

            frames = jnp.concatenate(jnp.stack(frames, 1))
            blocks = jnp.concatenate(jnp.stack(blocks, 1))
            return frames, states, blocks

        def step_env_render(carry, _):
            rng_r, obs_r, env_state_r, network_params = carry
            rng_r, _rng_r = jax.random.split(rng_r)
            
            pi, value = network.apply(network_params, obs_r)
            action_r = get_expert_action(env_state_r)#pi.sample(seed=rng_r)

            rng_step = jax.random.split(_rng_r, config.n_render_eps)

            # rng_step_r = rng_step_r.reshape((config.n_gpus, -1) + rng_step_r.shape[1:])
            vmap_step_fn = jax.vmap(env_r.step, in_axes=(0, 0, 0, None))
            # pmap_step_fn = jax.pmap(vmap_step_fn, in_axes=(0, 0, 0, None))
            obs_r, env_state_r, reward_r, done_r, info_r = vmap_step_fn(
                            rng_step, env_state_r, action_r,
                            env_params)
            vmap_render_fn = jax.vmap(env_r.render, in_axes=(0,))
            # pmap_render_fn = jax.pmap(vmap_render_fn, in_axes=(0,))
            frames = vmap_render_fn(env_state_r)
            # Get rid of the gpu dimension
            # frames = jnp.concatenate(jnp.stack(frames, 1))
            if config.env_name == 'Lego':
                vmap_block_fn = jax.vmap(env_r.get_blocks, in_axes = (0,))
                blocks = vmap_block_fn(env_state_r)

                return (rng_r, obs_r, env_state_r, network_params),\
                    (env_state_r, reward_r, done_r, info_r, frames, blocks)
            
            return (rng_r, obs_r, env_state_r, network_params),\
                    (env_state_r, reward_r, done_r, info_r, frames)

        
        
        frames, states, blocks = render_episodes(train_state.params)
        jax.debug.callback(render_frames, frames, runner_state.update_i, states)
        jax.debug.callback(render_mpds, blocks, runner_state.update_i, states)
        old_render_results = (frames, states, blocks)
        
        

       

    return lambda rng: train(rng, config)


def init_checkpointer(config: Config):
    # This will not affect training, just for initializing dummy env etc. to load checkpoint.
    rng = jax.random.PRNGKey(30)
    # Set up checkpointing
    ckpt_dir = get_ckpt_dir(config)

    # Create a dummy checkpoint so we can restore it to the correct dataclasses
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    # env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    rng, _rng = jax.random.split(rng)
    network = get_network(env, env_params, config)
    init_x = env.gen_dummy_obs(env_params)
    # init_x = env.observation_space(env_params).sample(_rng)[None, ]
    network_params = network.init(_rng, init_x)
    tx = optax.chain(
        optax.clip_by_global_norm(config.MAX_GRAD_NORM),
        optax.adam(config.lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.n_envs)

    # reset_rng_r = reset_rng.reshape((config.n_gpus, -1) + reset_rng.shape[1:])
    vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
    # pmap_reset_fn = jax.pmap(vmap_reset_fn, in_axes=(0, None))
    obsv, env_state = vmap_reset_fn(
        reset_rng, 
        env_params, 
        gen_dummy_queued_state(config, env, reset_rng)
    )
    runner_state = RunnerState(train_state=train_state, env_state=env_state, last_obs=obsv,
                               # ep_returns=jnp.full(config.num_envs, jnp.nan), 
                               rng=rng, update_i=0)
    target = {'runner_state': runner_state, 'step_i': 0}
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        ckpt_dir, orbax.checkpoint.PyTreeCheckpointer(), options)

    if checkpoint_manager.latest_step() is None:
        restored_ckpt = None
    else:
        steps_prev_complete = checkpoint_manager.latest_step()
        restored_ckpt = checkpoint_manager.restore(
            steps_prev_complete, items=target)
        restored_ckpt['steps_prev_complete'] = steps_prev_complete


    return checkpoint_manager, restored_ckpt

    
def gen_dummy_queued_state(config, env, frz_rng):
    queued_state = QueuedState(
        ctrl_trgs= None, #jnp.zeros(len(env.prob.stat_trgs)),
        frz_map=jnp.zeros(env.map_shape, dtype=bool)
    )
    return queued_state


@hydra.main(version_base=None, config_path='./', config_name='lego_pcgrl')
def main(config: TrainConfig):
    config.learning_mode = "IL"
    config = init_config(config, evo=False)
    rng = jax.random.PRNGKey(config.seed)

    exp_dir = config.exp_dir 


    # Need to do this before setting up checkpoint manager so that it doesn't refer to old checkpoints.
    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    checkpoint_manager, restored_ckpt = init_checkpointer(config)

    # if restored_ckpt is not None:
    #     ep_return s = restored_ckpt['runner_state'].ep_returns
    #     plot_ep_returns(ep_returns, config)
    # else:
    if restored_ckpt is None:
        progress_csv_path = os.path.join(exp_dir, "progress.csv")
        assert not os.path.exists(progress_csv_path), "Progress csv already exists, but have no checkpoint to restore " +\
            "from. Run with `overwrite=True` to delete the progress csv."
        # Create csv for logging progress
        with open(os.path.join(exp_dir, "progress.csv"), "w") as f:
            f.write("timestep,ep_return\n")

    

    train_jit = jax.jit(make_train(config, restored_ckpt, checkpoint_manager))
    out = train_jit(rng)

#   ep_returns = out["runner_state"].ep_returns


if __name__ == "__main__":
    main()