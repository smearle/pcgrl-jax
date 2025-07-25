import json
import os
import shutil
from timeit import default_timer as timer
from typing import NamedTuple
from gymnax import EnvState

import hydra
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import imageio
import optax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax
from tensorboardX import SummaryWriter

from conf.config import Config, TrainConfig
from envs.pcgrl_env import PCGRLObs, QueuedState, gen_static_tiles, render_stats
from evo_accel import EvoState, apply_evo, gen_discount_factors_matrix
from purejaxrl.experimental.s5.wrappers import LogWrapper
from utils import (get_ckpt_dir, get_exp_dir, init_network, gymnax_pcgrl_make,
                   init_config)


class RunnerState(struct.PyTreeNode):
    train_state: TrainState
    env_state: EnvState
    evo_state: EvoState
    last_obs: jnp.ndarray
    # rng_act: jnp.ndarray
#   ep_returns: jnp.ndarray
    rng: jnp.ndarray
    update_i: int
    last_eval_results: tuple


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    # rng_act: jnp.ndarray


def make_train(config: TrainConfig, restored_ckpt, checkpoint_manager):
    config._num_updates = (
        config.total_timesteps // config.num_steps // config.n_envs
    )
    config._minibatch_size = (
        config.n_envs * config.num_steps // config.NUM_MINIBATCHES
    )
    env_r, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env_e, env_params_e = gymnax_pcgrl_make(config.env_name, config=config)
    env_e = LogWrapper(env_e)
    # env = FlattenObservationWrapper(env)
    env = LogWrapper(env_r)  # Does this need to be a LogWrapper env? No(?)
    env_r.init_graphics()

    max_episode_steps = env._env.max_steps
    # Generating this here for efficiency since it will never change.
    # (Used in ACCEL value function error computation.)
    discount_factor_matrix = gen_discount_factors_matrix(
        config.GAMMA, max_episode_steps)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.NUM_MINIBATCHES * config.update_epochs))
            / config._num_updates
        )
        return config["LR"] * frac

    def train(rng, config: TrainConfig):
        # INIT NETWORK
        network = init_network(env, env_params, config)

        rng, _rng = jax.random.split(rng)
        init_x = env.gen_dummy_obs(env_params)
        # init_x = env.observation_space(env_params).sample(_rng)[None]
        network_params = network.init(_rng, init_x)
        print(network.subnet.tabulate(_rng, init_x.map_obs, init_x.flat_obs))
        # print(network.subnet.tabulate(_rng, init_x, jnp.zeros((init_x.shape[0], 0))))

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
        frz_rng = jax.random.split(_rng, config.evo_pop_size)
        frz_maps = jax.vmap(gen_static_tiles, in_axes=(0, None, None, None))(frz_rng, 0.1, 0, env.map_shape)
        evo_state = EvoState(frz_map=frz_maps, top_fitness=jnp.zeros(config.evo_pop_size))
        frz_maps = jnp.tile(frz_maps, (int(np.ceil(config.n_envs / config.evo_pop_size)), 1, 1))[:config.n_envs]
        queued_state = QueuedState(ctrl_trgs=jnp.zeros(len(env.prob.stat_trgs)))
        queued_state = jax.vmap(env.queue_frz_map, in_axes=(None, 0))(queued_state, frz_maps) 
        reset_rng = jax.random.split(_rng, config.n_envs)
        vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, 0))
        obsv, env_state = vmap_reset_fn(reset_rng, env_params, queued_state)

        # INIT ENV FOR EVAL
        rng_e, _rng_e = jax.random.split(rng)
        reset_rng_e = jax.random.split(_rng_e, config.n_envs) # maybe we don't need this new rng?
        # load the user defined map for evaluation
        with open(config.eval_map_path, 'r') as f:
            eval_maps = json.load(f)
        eval_maps = jnp.array(list(eval_maps.values())).astype(bool)
        # also create new random freezies for evaluation
        rng_ef, _rng_ef = jax.random.split(rng_e)
        frz_rng_ee = jax.random.split(_rng_ef, config.n_eval_maps)
        frz_maps_ee = jax.vmap(gen_static_tiles, in_axes=(0, None, None, None))(frz_rng_ee, 0.1, 0, env.map_shape)
        eval_maps = jnp.concatenate((eval_maps, frz_maps_ee), axis=0)

        eval_maps = jnp.tile(eval_maps, (int(np.ceil(config.n_envs / eval_maps.shape[0])), 1, 1))[:config.n_envs]
        queued_state_e = QueuedState(ctrl_trgs=jnp.zeros(len(env.prob.stat_trgs)))
        queued_state_e = jax.vmap(env.queue_frz_map, in_axes=(None, 0))(queued_state_e, eval_maps)
        vmap_reset_fn_e = jax.vmap(env_e.reset, in_axes=(0, None, 0))
        obsv_e, env_state_e = vmap_reset_fn_e(reset_rng_e, env_params_e, queued_state_e)

        # INIT ENV FOR RENDER
        rng_r, _rng_r = jax.random.split(rng)
        reset_rng_r = jax.random.split(_rng_r, config.n_render_eps)
        queued_state = QueuedState(ctrl_trgs=jnp.zeros(len(env.prob.stat_trgs)))
        queued_state = jax.vmap(env.queue_frz_map, in_axes=(None, 0))(queued_state, frz_maps[:config.n_render_eps])
        vmap_reset_fn = jax.vmap(env_r.reset, in_axes=(0, None, 0))
        # pmap_reset_fn = jax.pmap(vmap_reset_fn, in_axes=(0, None))
        obsv_r, env_state_r = vmap_reset_fn(reset_rng_r, env_params, queued_state)  # Replace None with your env_params if any
        
        # obsv_r, env_state_r = jax.vmap(
        #     env_r.reset, in_axes=(0, None))(reset_rng_r, env_params)

        rng, _rng = jax.random.split(rng)
#       ep_returns = jnp.full(shape=config.NUM_UPDATES,
#       ep_returns = jnp.full(shape=1,
#                             fill_value=jnp.nan, dtype=jnp.float32)
        steps_prev_complete = 0

        def step_env_eval(carry, _):
            rng_e, obs_e, env_state_e, network_params = carry
            rng_e, _rng_e = jax.random.split(rng_e)

            pi, value = network.apply(network_params, obs_e)
            action_e = pi.sample(seed=rng_e)
            
            rng_step = jax.random.split(_rng_e, config.n_envs)

            vmap_step_fn = jax.vmap(env_e.step, in_axes=(0, 0, 0, None))
            obs_e, env_state_e, reward_e, done_e, info_e = vmap_step_fn(
                            rng_step, env_state_e, action_e,
                            env_params)
            return (rng_e, obs_e, env_state_e, network_params),\
                (env_state_e, reward_e, done_e, info_e)

        def eval_episodes(network_params, queued_state):
            

           

            obsv_e, env_state_e = jax.vmap(env_e.reset, in_axes=(0, None, 0))(reset_rng_e, env_params_e, queued_state)
            _, (states, rewards, dones, infos) = jax.lax.scan(
                step_env_eval, (rng_e, obsv_e, env_state_e, network_params),
                None, 1*env.max_steps)
            
            return rewards, states    # TODO: maybe not only return rewards but also stats?

        rewards_e, eval_states = eval_episodes(
            train_state.params, env_state_e.env_state.queued_state)
        last_eval_results = (rewards_e, eval_states)

        runner_state = RunnerState(
            train_state=train_state, env_state=env_state, evo_state=evo_state,
            last_obs=obsv, rng=rng, update_i=0, last_eval_results=last_eval_results)

        # exp_dir = get_exp_dir(config)
        if restored_ckpt is not None:
            steps_prev_complete = restored_ckpt['steps_prev_complete']
            runner_state = restored_ckpt['runner_state']
            steps_remaining = config.total_timesteps - steps_prev_complete
            config._num_updates = int(
                steps_remaining // config.num_steps // config.n_envs)

            # TODO: Overwrite certain config values

        def render_frames(frames, i, env_states=None):
            if i % config.render_freq != 0:
            # if jnp.all(frames == 0):
                return
            print(f"Rendering episode gifs at update {i}")
            assert len(frames) == config.n_render_eps * 1 * env.max_steps,\
                "Not enough frames collected"

            if config.env_name == 'Candy':
                # Render intermediary frames.
                pass

            # Save gifs.
            for ep_is in range(config.n_render_eps):
                gif_name = f"{config.exp_dir}/update-{i}_ep-{ep_is}.gif"
                ep_frames = frames[ep_is*env.max_steps:(ep_is+1)*env.max_steps]

                # new_frames = []
                # for i, frame in enumerate(frames):
                #     state_i = jax.tree_util.tree_map(lambda x: x[i], env_states)
                #     frame = render_stats(env_r, state_i, frame)
                #     new_frames.append(frame)
                # frames = new_frames

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

        def render_episodes(network_params, queued_state):
            queued_state = jax.tree_map(lambda x: x[:config.n_render_eps], queued_state)
            obsv_r, env_state_r = jax.vmap(env_r.reset, in_axes=(0, None, 0))(
                reset_rng_r, env_params, queued_state
            )
            _, (states, rewards, dones, infos, frames) = jax.lax.scan(
                step_env_render, (rng_r, obsv_r, env_state_r, network_params),
                None, 1*env.max_steps)

            frames = jnp.concatenate(jnp.stack(frames, 1))
            return frames, states

        def step_env_render(carry, _):
            rng_r, obs_r, env_state_r, network_params = carry
            rng_r, _rng_r = jax.random.split(rng_r)

            pi, value = network.apply(network_params, obs_r)
            action_r = pi.sample(seed=rng_r)
            # action_r = jnp.full(action_r.shape, fill_value=0)

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

        # Dummy render to initialize the render frames
        frames, states = render_episodes(train_state.params, env_state.env_state.queued_state)
        jax.debug.callback(render_frames, frames, runner_state.update_i, states)
        old_render_results = (frames, states)

        # jax.debug.print(f'Rendering episode gifs took {timer() - start_time} seconds')

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            last_eval_results = runner_state.last_eval_results
            # COLLECT TRAJECTORIES
            def _env_step(runner_state: RunnerState, unused):
                train_state, env_state, evo_state, last_obs, rng, update_i = (
                    runner_state.train_state, runner_state.env_state,
                    runner_state.evo_state,
                    runner_state.last_obs,
                    runner_state.rng, runner_state.update_i,
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                # Squash the gpu dimension (network only takes one batch dimension)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                # action = jnp.full(action.shape, 0) # FIXME DUMDUM Only for cleaning all blocks (debugging evo)
                log_prob = pi.log_prob(action)

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
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = RunnerState(
                    train_state, env_state, evo_state, obsv, rng,
                    update_i=update_i, last_eval_results=runner_state.last_eval_results)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, evo_state, last_obs, rng = \
                runner_state.train_state, runner_state.env_state, \
                runner_state.evo_state, runner_state.last_obs, runner_state.rng
            
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config.GAMMA * \
                        next_value * (1 - done) - value
                    gae = (
                        delta
                        + config.GAMMA * config.GAE_LAMBDA * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        # obs = traj_batch.obs[None]
                        pi, value = network.apply(params, traj_batch.obs)
                        # action = traj_batch.action.reshape(pi.logits.shape[:-1])
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.CLIP_EPS, config.CLIP_EPS)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses,
                                              value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        # Some reshaping to accomodate player, x, and y 
                        # dimensions to action output. (Not used often...)
                        gae = gae[..., None, None, None]

                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.CLIP_EPS,
                                1.0 + config.CLIP_EPS,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.VF_COEF * value_loss
                            - config.ENT_COEF * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = \
                    update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config._minibatch_size * config.NUM_MINIBATCHES
                assert (
                    batch_size == config.num_steps * config.n_envs
                ), "batch size must be equal to number of steps * number " + \
                    "of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
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
                update_state = (train_state, traj_batch,
                                advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
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

            # eval the frz maps and mutate the frz maps
            # NOTE: If you vmap the train function, both of these branches will (most probably)
            # be evaluated each time.                  
            frz_maps = env_state.env_state.queued_state.frz_map
            evo_state: EvoState = jax.lax.cond(
                runner_state.update_i % config.evo_freq == 0,
                lambda: apply_evo(
                    rng, frz_maps, env, env_params, 
                    network_params=network_params, network=network,
                    config=config,
                    discount_factor_matrix=discount_factor_matrix),
                lambda: evo_state)
            
            frz_maps = evo_state.frz_map
            frz_maps = jnp.tile(frz_maps, (int(np.ceil(config.n_envs / config.evo_pop_size)), 1, 1))[:config.n_envs]
            queued_state = QueuedState(ctrl_trgs=jnp.zeros(len(env.prob.stat_trgs)))
            queued_state = jax.vmap(env.queue_frz_map, in_axes=(None, 0))(queued_state, frz_maps)
            env_state = env_state.replace(env_state=env_state.env_state.replace(queued_state=queued_state))
            
            # Create a tensorboard writer
            writer = SummaryWriter(get_exp_dir(config))

            def log_callback(metric, evo_fits, eval_rewards, steps_prev_complete):
                timesteps = metric["timestep"][metric["returned_episode"]
                                               ] * config.n_envs
                if len(timesteps) > 0:
                    t = timesteps[0] + steps_prev_complete
                    ep_return = (metric["returned_episode_returns"]
                                 [metric["returned_episode"]].mean(
                    ))
                    ep_length = (metric["returned_episode_lengths"]
                                  [metric["returned_episode"]].mean())

                    # Add a row to csv with ep_return
                    with open(os.path.join(get_exp_dir(config),
                                           "progress.csv"), "a") as f:
                        f.write(f"{t},{ep_return}\n")

                    writer.add_scalar("ep_return", ep_return, t)
                    writer.add_scalar("ep_length", ep_length, t)
                    writer.add_scalar("mean_fitness", evo_fits.mean(), t)
                    writer.add_scalar("eval_rewards", eval_rewards.mean(), t)
                    # for k, v in zip(env.prob.metric_names, env.prob.stats):
                    #     writer.add_scalar(k, v, t)

            # FIXME: shouldn't assume size of render map.
            frames_shape = (config.n_render_eps * 1 * env.max_steps, 
                            env.tile_size * (env.map_shape[0] + 2),
                            env.tile_size * (env.map_shape[1] + 2), 4)

            # FIXME: Inside vmap, both conditions will be executed. But that's ok for now
            #   since we don't vmap `train`.
            frames, states = jax.lax.cond(
                runner_state.update_i % config.render_freq == 0,
                lambda: render_episodes(train_state.params, queued_state),
                lambda: old_render_results,)
            jax.debug.callback(render_frames, frames, runner_state.update_i, states)
            # jax.debug.print(f'Rendering episode gifs took {timer() - start_time} seconds')

            eval_rewards, states = jax.lax.cond(
                runner_state.update_i % config.eval_freq == 0,
                lambda: eval_episodes(train_state.params, queued_state_e),
                lambda: last_eval_results,)

            last_eval_results = (eval_rewards, states)
            eval_rewards = eval_rewards.mean(0)
            evo_fits = evo_state.top_fitness

            jax.debug.callback(log_callback, metric, evo_fits, eval_rewards,
                               steps_prev_complete)

            runner_state = RunnerState(
                train_state=train_state, env_state=env_state, evo_state=evo_state,
                last_obs=last_obs, rng=rng,
                update_i=runner_state.update_i+1,
                last_eval_results=last_eval_results)

            return runner_state, metric

        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config._num_updates
        )

        jax.debug.callback(save_checkpoint, runner_state,
                           metric, steps_prev_complete)

        return {"runner_state": runner_state, "metrics": metric}

    return lambda rng: train(rng, config)


# def plot_ep_returns(ep_returns, config):
#     plt.plot(ep_returns)
#     plt.xlabel("Timesteps")
#     plt.ylabel("Episodic Return")
#     plt.title(f"Episodic Return vs. Timesteps ({config.ENV_NAME})")
#     plt.savefig(os.path.join(get_exp_dir(config), "ep_returns.png"))


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
    network = init_network(env, env_params, config)
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

    frz_rng = jax.random.split(_rng, config.evo_pop_size)
    frz_maps = jax.vmap(gen_static_tiles, in_axes=(0, None, None, None))(frz_rng, 0.1, 0, env.map_shape)
    evo_state = EvoState(frz_map=frz_maps, top_fitness=jnp.zeros(config.evo_pop_size))
    frz_maps = jnp.tile(frz_maps, (config.n_envs // config.evo_pop_size, 1, 1))

    queued_state = QueuedState(ctrl_trgs=jnp.zeros(len(env.prob.stat_trgs)))
    queued_state = jax.vmap(env.queue_frz_map, in_axes=(None, 0))(queued_state, frz_maps)

    # reset_rng_r = reset_rng.reshape((config.n_gpus, -1) + reset_rng.shape[1:])
    vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, 0))
    # pmap_reset_fn = jax.pmap(vmap_reset_fn, in_axes=(0, None))
    obsv, env_state = vmap_reset_fn(reset_rng, env_params, queued_state)
    dummy_last_eval_results = (jnp.zeros((env._env.max_steps, config.n_envs)), env_state)
    runner_state = RunnerState(train_state=train_state, env_state=env_state, evo_state=evo_state,
                               last_obs=obsv,
                               # ep_returns=jnp.full(config.num_envs, jnp.nan), 
                               rng=rng, update_i=0, last_eval_results=dummy_last_eval_results)
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

        # # Load the csv as a dataframe and delete all rows after the last checkpoint
        # progress_csv_path = os.path.join(get_exp_dir(config), "progress.csv")
        # progress_df = pd.read_csv(progress_csv_path, names=["timestep", "ep_return"])
        # # Convert timestep to int

        # progress_df = progress_df[progress_df["timestep"] <= steps_prev_complete]
        # progress_df.to_csv(progress_csv_path, header=False, index=False)

    return checkpoint_manager, restored_ckpt


@hydra.main(version_base="1.3", config_path='./conf', config_name='train_accel')
def main(config: TrainConfig):
    config = init_config(config)
    rng = jax.random.PRNGKey(config.seed)

    exp_dir = config.exp_dir

    # Need to do this before setting up checkpoint manager so that it doesn't refer to old checkpoints.
    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    checkpoint_manager, restored_ckpt = init_checkpointer(config)

    # if restored_ckpt is not None:
    #     ep_returns = restored_ckpt['runner_state'].ep_returns
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
