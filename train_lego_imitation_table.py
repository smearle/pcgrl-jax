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
from envs.probs.lego import LegoProblemState, tileNames, tileDims, LegoMetrics
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

def get_expert_action(log_env_state):
    env_state = log_env_state.env_state
    moves = jnp.array([
            (0,0), #no move, goes to top
            (0,1),
            (0,-1),
            (1,0),
            (-1,0),
            (0,0) # no move, does not go to top  
        ])
    
    blocks = env_state.rep_state.blocks
    curr_block_inds = env_state.rep_state.curr_block
    env_map = env_state.env_map

    curr_blocks = jnp.array([blocks[i, curr_block_inds[i], :] for i in range(blocks.shape[0])])
    curr_x = curr_blocks[:,0]
    curr_z = curr_blocks[:,2]
    curr_y = curr_blocks[:,1]
    curr_block_type = curr_blocks[:,3] 
    
    roof_x = blocks[:,-1,0]
    roof_z = blocks[:,-1,2]

    table_mask = jnp.zeros((4, env_map.shape[2], 4)).at[[0, -1], :, [0, -1]].set(1)
    table_mask = table_mask.at[0,:,-1].set(1).at[-1,:,0].set(1)
    #test = jnp.count_nonzero(table_mask, axis=1)

    def slice_map(carry, args):
        env_map_i, roof_x_i, roof_z_i = args
        sliced= jax.lax.dynamic_slice(env_map_i, start_indices=(roof_x_i, 0, roof_z_i), slice_sizes=(4, env_map_i.shape[1], 4))
        masked_slice = jnp.where(table_mask==0, 0, sliced)
        table_map = jax.lax.dynamic_update_slice(jnp.zeros(env_map.shape[1:]), masked_slice, (roof_x_i, 0, roof_z_i))  
        return None, table_map

    _, table_map = jax.lax.scan(slice_map, None, (env_map, roof_x, roof_z))

    leg_heights = jnp.count_nonzero(table_map, axis=2)
    leg_mask = leg_heights != 0
    leg_heights_adjusted = jnp.where(leg_mask, leg_heights, jnp.inf)
    min_leg_hts = jnp.argmin(leg_heights_adjusted.reshape(leg_heights.shape[0], -1), axis=1)
    min_leg_hts_stacked = jnp.stack((min_leg_hts//leg_heights.shape[2], min_leg_hts%leg_heights.shape[2]), axis=1)

    legs_x_pos = min_leg_hts_stacked[:,0]
    legs_z_pos = min_leg_hts_stacked[:,1]
        
    x_dir = legs_x_pos-curr_x
    z_dir = legs_z_pos-curr_z

    #when block is large flat, don't move except to top
    x_dir = jnp.where(curr_block_type == 3, 0, x_dir)
    z_dir = jnp.where(curr_block_type == 3, 0, z_dir)

    #when block is already in a leg, don't move
    truth_cond = jnp.logical_and(
        jnp.logical_and(
            jnp.logical_or(
                curr_x == roof_x,
                curr_x == roof_x + 3
            ),
            jnp.logical_or(
                curr_z == roof_z,
                curr_z == roof_z + 3
            )
        ),
        jnp.logical_and(
            curr_block_type != 3,
            curr_y < 3*(blocks.shape[1]//4)
        ) 
        )  

    x_moves = jnp.clip(x_dir, a_min = -1, a_max = 1).reshape(x_dir.shape[0], 1)
    z_moves = jnp.clip(z_dir, a_min = -1, a_max = 1).reshape(x_dir.shape[0], 1)
    z_moves = jnp.where(x_moves==0, z_moves, 0)
    combined_actions = jnp.concatenate((x_moves, z_moves), axis=1)

    def find_indices(array1, array2):
        # Step 1: Compare each row of array1 with each row of array2
        comparison = array1[:, None, :] == array2[None, :, :]  # Shape becomes (4, 5, 2)
        # Step 2: Check where all elements along the last axis match
        matches = jnp.all(comparison, axis=2)  # Shape becomes (4, 5)
        # Step 3: Find the index in the second array for each row of the first arrayl
        indices = jnp.argmax(matches, axis=1)  # Shape becomes (4,)

        return indices

    tmp_actions = find_indices(combined_actions, moves)
    
    actions = jnp.where(truth_cond, 5, tmp_actions)#action 0 where a table leg is already in place
    
    #jax.debug.breakpoint()
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
            is_finished = states.done
            

            for ep_is in range(config.n_render_eps):
                dones = is_finished[:,ep_is]
                done_ind = jnp.argmax(dones)
                done_ind = jax.lax.cond(done_ind == 0, lambda: env.max_steps-1, lambda: done_ind)

                ep_blocks = blocks[ep_is*env.max_steps:(ep_is+1)*env.max_steps]#[:2]

                ep_end_avg_height = states.prob_state.stats[done_ind-1, ep_is, LegoMetrics.AVG_HEIGHT]
                ep_footprint = states.prob_state.stats[done_ind-1, ep_is, LegoMetrics.FOOTPRINT]
                ep_cntr_dist = states.prob_state.stats[done_ind-1, ep_is, LegoMetrics.CENTER]
                ep_curr_blocks = states.rep_state.curr_block[:,ep_is]
                actions = states.rep_state.last_action[:,ep_is]
                tableness = states.prob_state.stats[:, ep_is, LegoMetrics.TABLE]
                max_tableness = jnp.max(tableness)

                
                savedir = f"{config.exp_dir}/mpds/update-{i}_ep{ep_is}_ht{ep_end_avg_height:.2f}_ctrdist{ep_cntr_dist:.2f}_tableness{max_tableness:.2f}/"
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                
                for num in range(ep_blocks.shape[0]):
                    curr_blocks = ep_blocks[num,:,:]
                    curr_block = ep_curr_blocks[num]
                    action = actions[num]


                    savename = os.path.join(savedir, f"{num}_a{action}_table{tableness[num]:.2f}.mpd")
                    
                    f = open(savename, "a")
                    f.write("0/n")
                    f.write("0 Name: New Model.ldr\n")
                    f.write("0 Author:\n")
                    f.write("\n")

                    for x in range(config.map_width):
                        for z in range(config.map_width):
                            lego_block_name = "3005"
                            block_color = "2 "
        
                            y_offset = -3#-24

                            x_lego = x * 20 + 10
                            y_lego =  0#(1)*(LegoDimsDict[lego_block_name][1])
                            z_lego = z * 20 + 10
                            
                            f.write("1 ")
                            f.write(block_color)
                            f.write(str(x_lego) + ' ' + str(y_lego) + ' ' + str(z_lego) + ' ')
                            f.write("1 0 0 0 1 0 0 0 1 ")
                            f.write(lego_block_name + ".dat")
                            f.write("\n")

                    
                    y_offset = -24 #1 brick height (plate is 8)
           
                    for b in range(curr_blocks.shape[0]):
                        blocktype = curr_blocks[b,3]
                        lego_block_name = tileNames[blocktype]
                        block_color = "7 "
                        if curr_block == b:
                                block_color = "14 "
                        if b == (curr_block-1)%curr_blocks.shape[0]:
                            block_color = "46 "
                        x_lego = curr_blocks[b, 2] * 20 + 10*(tileDims[blocktype][2])
                        y_lego = curr_blocks[b, 1] * (y_offset)/3 + (y_offset/3)*(tileDims[blocktype][1])
                        z_lego = curr_blocks[b, 0] * 20 + 10*(tileDims[blocktype][0])
                        
                        f.write("1 ")
                        f.write(block_color)
                        f.write(str(x_lego) + ' ' + str(y_lego) + ' ' + str(z_lego) + ' ')
                        f.write("1 0 0 0 1 0 0 0 1 ") #orientation matrix stays set for now. TO DO: block rotation
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
            action_r = pi.sample(seed=rng_r)

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

      

        def save_checkpoint(runner_state, info, steps_prev_complete):
            try:
                timesteps = info["timestep"][info["returned_episode"]
                                             ] * config.n_envs
            except jax.errors.NonConcreteBooleanIndexError:
                return
            for t in timesteps:
                if t > 0:
                    latest_ckpt_step = checkpoint_manager.latest_step()
                    if (latest_ckpt_step is None or
                            t - latest_ckpt_step >= config.ckpt_freq):
                        print(f"Saving checkpoint at step {t}")
                        ckpt = {'runner_state': runner_state,
                                'config': config, 'step_i': t}
                        # ckpt = {'step_i': t}
                        save_args = orbax_utils.save_args_from_target(ckpt)
                        checkpoint_manager.save(t, ckpt, save_kwargs={
                                                'save_args': save_args})
                    break
        
        frames, states, blocks = render_episodes(train_state.params)
        jax.debug.callback(render_frames, frames, runner_state.update_i, states)
        jax.debug.callback(render_mpds, blocks, runner_state.update_i, states)
        old_render_results = (frames, states, blocks)
        
        def _update_step(runner_state, batch):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state: RunnerState, unused):
                train_state, env_state, last_obs, rng, update_i = (
                    runner_state.train_state, runner_state.env_state,
                    runner_state.last_obs,
                    runner_state.rng, runner_state.update_i,
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                # Squash the gpu dimension (network only takes one batch dimension)
                pi, _ = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                #GET EXPERT ACTION
                expert_action = get_expert_action(env_state)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.n_envs)

                # rng_step = rng_step.reshape((config.n_gpus, -1) + rng_step.shape[1:])
                vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))
                # pmap_step_fn = jax.pmap(vmap_step_fn, in_axes=(0, 0, 0, None))
                obsv, env_state, reward, done, info = vmap_step_fn(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(
                    done, action, expert_action, reward, log_prob, last_obs, info
                )
                runner_state = RunnerState(
                    train_state, env_state, obsv, rng,
                    update_i=update_i)
                return runner_state, transition
            
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.max_steps_multiple*config.n_blocks
            )

            train_state, env_state, last_obs, rng = runner_state.train_state, runner_state.env_state, \
                runner_state.last_obs, runner_state.rng

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch):
                    traj_batch = batch

                    def _loss_fn(params, traj_batch):
                        # RERUN  
                        pi, _ = network.apply(params, traj_batch.obs)

                        probs = pi.log_prob(traj_batch.expert_action)
                        logits = pi.logits
                        #jax.debug.print("min logits: {l}", l=jnp.min(logits))
                        #jax.debug.print("max logits: {l}", l=jnp.max(logits))
                        #jax.debug.print("logits 0: {l}", l=logits[0])
                        #jax.debug.print("logits 1: {l}", l=logits[1])
                        #jax.debug.print("expert_action 0: {l}", l=traj_batch.expert_action[0])
                        #jax.debug.print("expert_action 1: {l}", l=traj_batch.expert_action[1])
                        #jax.debug.print("probabilities: {p}", p=jnp.exp(logits[0] - jax.scipy.special.logsumexp(logits[0])))
                        
                       
                        probs_scaled = probs/probs.size
                        loss = -jnp.sum(probs_scaled)
                        #jax.debug.print("min probs: {p}", p=jnp.min(probs))
                        #jax.debug.print("max probs: {p}", p=jnp.max(probs))
                        #jax.debug.print("log_probs 0: {l}", l=probs[0])
                        #jax.debug.print("log_probs 1: {l}", l=probs[1])

                        #jax.debug.print("loss: {l}", l=loss)

                        return loss
                    
                    
                    #def cross_entropy_loss_fn(params, traj_batch):
                    #def _loss_fn(params, traj_batch):
                    #   logits = network.apply(params, traj_batch.obs)
                    #   log_softmax_logits = jax.nn.log_softmax(logits[1])
                    #   loss = -jnp.mean(log_softmax_logits * traj_batch.expert_action)
                    #   return loss

                    #def kl_divergence_loss_fn(params, traj_batch):
                    #    pi, _ = network.apply(params, traj_batch.obs)
                    #    expert_pi = some_expert_distribution(traj_batch)
                    #    loss = jnp.mean(jax.scipy.stats.entropy(pi, expert_pi))
                    #    return loss

                    
                    
                    grad_fn = jax.value_and_grad(_loss_fn)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, rng = \
                    update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
                assert (
                    batch_size == config.n_blocks * config.max_steps_multiple * config.n_envs
                ), "batch size must be equal to number of steps * number " + \
                    "of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = traj_batch
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch,rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            traj_batch = update_state[1]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.DEBUG:
                def callback(info, steps_prev_complete):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]
                                                 ] * config.n_envs
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, metric, steps_prev_complete)

            jax.debug.callback(save_checkpoint, runner_state,
                               metric, steps_prev_complete)

            # Create a tensorboard writer
            writer = SummaryWriter(get_exp_dir(config))

            def log_callback(metric, steps_prev_complete, train_start_time, loss_info):
                timesteps = metric["timestep"][metric["returned_episode"]] * config.n_envs

                
                if len(timesteps) > 0:
                    t = timesteps[0]
                    ep_return = (metric["returned_episode_returns"]
                                 [metric["returned_episode"]].mean()
                                 )
                    losses = loss_info.mean()

                    ep_footprint = metric["stats"][metric["returned_episode"]][:,1].mean()
                    ep_avg_height = metric["stats"][metric["returned_episode"]][:,0].mean()
                    ep_ctr_dist = metric["stats"][metric["returned_episode"]][:,3].mean()
                    ep_length = metric["step"][metric["returned_episode"]].mean()+1

                    n_envs = metric["last_action"].shape[1]
                    mean_num_actions = sum([len(set(metric["last_action"][:,i])) for i in range(n_envs)])/n_envs
                    # Add a row to csv with ep_return
                    with open(os.path.join(get_exp_dir(config),
                                           "progress.csv"), "a") as f:
                        f.write(f"{t},{ep_return}\n")

                    writer.add_scalar("ep_return", ep_return, t)
                    writer.add_scalar("ep_stats/ep_length", ep_length, t)
                    writer.add_scalar("ep_stats/ep_end_footprint", ep_footprint,t)
                    writer.add_scalar("ep_stats/ep_end_avg_height", ep_avg_height, t)
                    writer.add_scalar("ep_stats/num_actions", mean_num_actions, t)
                    writer.add_scalar("ep_stats/ep_end_dist_ctr", ep_ctr_dist,t)
                    # for k, v in zip(env.prob.metric_names, env.prob.stats):
                    #     writer.add_scalar(k, v, t)
                    fps = (t - steps_prev_complete) / (timer() - train_start_time)
                    writer.add_scalar("fps", fps, t)
                    writer.add_scalar("loss", losses, t)


            # FIXME: Inside vmap, both conditions are likely to get executed. Any way around this?
            # Currently not vmapping the train loop though, so it's ok.
            # start_time = timer()
            frames, states, blocks = jax.lax.cond(
                runner_state.update_i % config.render_freq == 0,
                lambda: render_episodes(train_state.params),
                lambda: old_render_results,)
            jax.debug.callback(render_frames, frames, runner_state.update_i, states)
            jax.debug.callback(render_mpds, blocks, runner_state.update_i, states)
            # jax.debug.print(f'Rendering episode gifs took {timer() - start_time} seconds')

            jax.debug.callback(log_callback, metric,
                               steps_prev_complete, train_start_time, loss_info)

            runner_state = RunnerState(
                train_state, env_state, last_obs, rng,
                update_i=runner_state.update_i+1)

            return runner_state, metric

        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config.NUM_UPDATES
        )

        jax.debug.callback(save_checkpoint, runner_state,
                           metric, steps_prev_complete)

        return {"runner_state": runner_state, "metrics": metric}

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
    config.reward = ("TABLE",)
    config.render_freq = 50
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
