from functools import partial
import os
import shutil
from timeit import default_timer as timer
from typing import Any, NamedTuple, Tuple

import hydra
import jax
import jax.numpy as jnp
from flax import struct
import imageio
from omegaconf import OmegaConf
import optax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter

from conf.config import Config, TrainConfig
from envs.pcgrl_env import (gen_dummy_queued_state, gen_dummy_queued_state_old,
                            OldQueuedState)
from purejaxrl.experimental.s5.wrappers import LogWrapper
from utils import (get_ckpt_dir, get_exp_dir, init_network, gymnax_pcgrl_make,
                   init_config)


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
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    # rng_act: jnp.ndarray


def log_callback(metric, steps_prev_complete, config: Config, writer, train_start_time):
    timesteps = metric["timestep"][metric["returned_episode"]] * config.n_envs * config.num_steps
    return_values = metric["returned_episode_returns"][metric["returned_episode"]]

    # for t in range(len(timesteps)):
    #     print(
    #         f"global step={timesteps[t]}, episodic return={return_values[t]}")

    if len(timesteps) > 0:
        t = timesteps[-1].item()
        ep_return_mean = return_values.mean()
        ep_return_max = return_values.max()
        ep_return_min = return_values.min()
        print(f"global step={t}; episodic return mean: {ep_return_mean} " + \
            f"max: {ep_return_max}, min: {ep_return_min}")
        ep_length = (metric["returned_episode_lengths"]
                        [metric["returned_episode"]].mean())

        # Add a row to csv with ep_return
        with open(os.path.join(get_exp_dir(config),
                                "progress.csv"), "a") as f:
            f.write(f"{t},{ep_return_mean}\n")

        writer.add_scalar("ep_return", ep_return_mean, t)
        writer.add_scalar("ep_return_max", ep_return_max, t)
        writer.add_scalar("ep_return_min", ep_return_min, t)
        writer.add_scalar("ep_length", ep_length, t)
        fps = (t - steps_prev_complete) / (timer() - train_start_time)
        writer.add_scalar("fps", fps, t)

        print(f"fps: {fps}")
        # for k, v in zip(env.prob.metric_names, env.prob.stats):
        #     writer.add_scalar(k, v, t)


def make_train(config: TrainConfig, restored_ckpt, checkpoint_manager):
    config.NUM_UPDATES = (
        config.total_timesteps // config.num_steps // config.n_envs
    )
    config.MINIBATCH_SIZE = (
        config.n_envs * config.num_steps // config.NUM_MINIBATCHES
    )
    env_r, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    # env = FlattenObservationWrapper(env)
    env = LogWrapper(env_r)
    env_r.init_graphics()

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.NUM_MINIBATCHES * config.update_epochs))
            / config.NUM_UPDATES
        )
        return config["LR"] * frac

    def train(rng, config: TrainConfig):

        train_start_time = timer()

        # Create a tensorboard writer
        writer = SummaryWriter(get_exp_dir(config))

        # INIT NETWORK
        network = init_network(env, env_params, config)

        rng, _rng = jax.random.split(rng)
        init_x = env.gen_dummy_obs(env_params)
        # init_x = env.observation_space(env_params).sample(_rng)[None]
        network_params = network.init(_rng, init_x)

        # Print network architecture and number of learnable parameters
        print(network.subnet.tabulate(_rng, init_x.map_obs, init_x.flat_obs))
        # print(network.subnet.tabulate(_rng, init_x, jnp.zeros((init_x.shape[0], 0))))

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
        # obsv, env_state = jax.vmap(
        #     env.reset, in_axes=(0, None))(reset_rng, env_params)

        # Reshape reset_rng and other per-environment states to (n_devices, -1, ...)
        # reset_rng = reset_rng.reshape((config.n_gpus, -1) + reset_rng.shape[1:])

        dummy_queued_state = gen_dummy_queued_state(env)

        # Apply pmap
        vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
        # pmap_reset_fn = jax.pmap(vmap_reset_fn, in_axes=(0, None))
        obsv, env_state = vmap_reset_fn(reset_rng, env_params, dummy_queued_state)

        # INIT ENV FOR RENDER
        rng_r, _rng_r = jax.random.split(rng)
        reset_rng_r = jax.random.split(_rng_r, config.n_render_eps)

        # Apply pmap
        # reset_rng_r = reset_rng_r.reshape((config.n_gpus, -1) + reset_rng_r.shape[1:])
        vmap_reset_fn = jax.vmap(env_r.reset, in_axes=(0, None, None))
        # pmap_reset_fn = jax.pmap(vmap_reset_fn, in_axes=(0, None))
        obsv_r, env_state_r = vmap_reset_fn(reset_rng_r, env_params, dummy_queued_state)
        
        # obsv_r, env_state_r = jax.vmap(
        #     env_r.reset, in_axes=(0, None))(reset_rng_r, env_params)

        rng, _rng = jax.random.split(rng)
#       ep_returns = jnp.full(shape=config.NUM_UPDATES,
#       ep_returns = jnp.full(shape=1,
#                             fill_value=jnp.nan, dtype=jnp.float32)
        steps_prev_complete = 0
        runner_state = RunnerState(
            train_state, env_state, obsv, rng,
            update_i=0)

        # exp_dir = get_exp_dir(config)
        if restored_ckpt is not None:
            steps_prev_complete = restored_ckpt['steps_prev_complete']
            runner_state = restored_ckpt['runner_state']
            steps_remaining = config.total_timesteps - steps_prev_complete
            config.NUM_UPDATES = int(
                steps_remaining // config.num_steps // config.n_envs)

            # TODO: Overwrite certain config values

        _log_callback = partial(log_callback, config=config, writer=writer,
                               train_start_time=train_start_time,
                               steps_prev_complete=steps_prev_complete)

        # FIXME: Temporary hack for reloading binary after change to 
        #   agent_coords generation.
        if config.representation == 'narrow':
            runner_state = runner_state.replace(
                env_state=runner_state.env_state.replace(
                    env_state=runner_state.env_state.env_state.replace(
                        rep_state=runner_state.env_state.env_state.rep_state.replace(
                            agent_coords=runner_state.env_state.env_state.rep_state.agent_coords[:, :config.map_width**2]
                        )
                    )
                )
            )

        def render_frames(frames, i, metric):
            timesteps = metric["timestep"][metric["returned_episode"]
                                    ] * config.n_envs
            if len(timesteps) > 0:
                t = timesteps[0]
            else:
                t = 0
            if config.render_freq <= 0 or i % config.render_freq != 0 or t == steps_prev_complete:
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

        def render_episodes(network_params):
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
            # Get the global env timestep numbers corresponding to the points at which different episodes were finished
            timesteps = info["timestep"][info["returned_episode"]] * config.n_envs

            if len(timesteps) > 0:
                # Get the latest global timestep at which some episode was finished
                t = timesteps[-1].item()
                latest_ckpt_step = checkpoint_manager.latest_step()
                if (latest_ckpt_step is None or
                        t - latest_ckpt_step >= config.ckpt_freq):
                    print(f"Saving checkpoint at step {t}")
                    ckpt = {'runner_state': runner_state,
                            # 'config': OmegaConf.to_container(config),
                            'step_i': t}
                    # ckpt = {'step_i': t}
                    # save_args = orbax_utils.save_args_from_target(ckpt)
                    # checkpoint_manager.save(t, ckpt, save_kwargs={
                    #                         'save_args': save_args})
                    checkpoint_manager.save(t, args=ocp.args.StandardSave(ckpt))

        frames, states = render_episodes(train_state.params)
        # jax.debug.callback(render_frames, frames, runner_state.update_i, metric)
        old_render_results = (frames, states)

        # jax.debug.print(f'Rendering episode gifs took {timer() - start_time} seconds')

        # TRAIN LOOP
        def _update_step(runner_state, unused):
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
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
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
                    train_state, env_state, obsv, rng,
                    update_i=update_i)
                return runner_state, transition

            # DO ROLLOUTS
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state.train_state, runner_state.env_state, \
                runner_state.last_obs, runner_state.rng
            
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

                        # Some reshaping to accomodate player, x, and y dimensions to action output. (Not used often...)
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
                batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
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

            jax.debug.callback(save_checkpoint, runner_state,
                               metric, steps_prev_complete)

            # FIXME: shouldn't assume size of render map.
            # frames_shape = (config.n_render_eps * 1 * env.max_steps, 
            #                 env.tile_size * (env.map_shape[0] + 2),
            #                 env.tile_size * (env.map_shape[1] + 2), 4)

            # FIXME: Inside vmap, both conditions are likely to get executed. Any way around this?
            # Currently not vmapping the train loop though, so it's ok.
            # start_time = timer()
            should_render = runner_state.update_i % config.render_freq == 0
            frames, states = jax.lax.cond(
                should_render,
                lambda: render_episodes(train_state.params),
                lambda: old_render_results,)
            jax.lax.cond(
                should_render,
                partial(jax.debug.callback, render_frames),
                lambda _, __, ___: None,
                frames, runner_state.update_i, metric
            )
            # jax.debug.print(f'Rendering episode gifs took {timer() - start_time} seconds')

            jax.debug.callback(_log_callback, metric)

            runner_state = RunnerState(
                train_state, env_state, last_obs, rng,
                update_i=runner_state.update_i+1)

            return runner_state, metric

        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config.NUM_UPDATES
        )

        # One final logging/checkpointing call to ensure things finish off
        # neatly.
        jax.debug.callback(_log_callback, metric)
        jax.debug.callback(save_checkpoint, runner_state, metric,
                           steps_prev_complete)

        return {"runner_state": runner_state, "metrics": metric}

    return lambda rng: train(rng, config)


# def plot_ep_returns(ep_returns, config):
#     plt.plot(ep_returns)
#     plt.xlabel("Timesteps")
#     plt.ylabel("Episodic Return")
#     plt.title(f"Episodic Return vs. Timesteps ({config.ENV_NAME})")
#     plt.savefig(os.path.join(get_exp_dir(config), "ep_returns.png"))


def init_checkpointer(config: Config) -> Tuple[Any, dict]:
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
    network_params = network.init(_rng, x=init_x)
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
        gen_dummy_queued_state(env)
    )
    runner_state = RunnerState(train_state=train_state, env_state=env_state, last_obs=obsv,
                               # ep_returns=jnp.full(config.num_envs, jnp.nan), 
                               rng=rng, update_i=0)
    target = {'runner_state': runner_state, 'step_i': 0}
    # Get absolute path
    ckpt_dir = os.path.abspath(ckpt_dir)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=2, create=True)
    # checkpoint_manager = orbax.checkpoint.CheckpointManager(
    #     ckpt_dir, orbax.checkpoint.PyTreeCheckpointer(), options)
    checkpoint_manager = ocp.CheckpointManager(
        # ocp.test_utils.erase_and_create_empty(ckpt_dir),
        ckpt_dir,
        options=options,
    )

    def try_load_ckpt(steps_prev_complete, target):

        runner_state = target['runner_state']
        try:
            restored_ckpt = checkpoint_manager.restore(
                # steps_prev_complete, items=target)
                steps_prev_complete, args=ocp.args.StandardRestore(target))
        except KeyError:
            # HACK
            runner_state = runner_state.replace(
                env_state=runner_state.env_state.replace(
                    env_state=runner_state.env_state.env_state.replace(
                                queued_state=gen_dummy_queued_state_old(env)
                    )
                )
            )     
            target = {'runner_state': runner_state, 'step_i': 0}
            restored_ckpt = checkpoint_manager.restore(
                steps_prev_complete, items=target)
            
            
        restored_ckpt['steps_prev_complete'] = steps_prev_complete
        if restored_ckpt is None:
            raise TypeError("Restored checkpoint is None")

        # HACK
        if isinstance(runner_state.env_state.env_state.queued_state, OldQueuedState):
            dummy_queued_state = gen_dummy_queued_state(env)

            # Now add leading dimension with sizeto match the shape of the original queued_state
            dummy_queued_state = jax.tree_map(lambda x: jnp.array(x, dtype=bool) if isinstance(x, bool) else x, dummy_queued_state)
            dummy_queued_state = jax.tree_map(lambda x: jnp.repeat(x[None], config.n_envs, axis=0), dummy_queued_state)
            
            runner_state = restored_ckpt['runner_state']
            runner_state = runner_state.replace(
                env_state=runner_state.env_state.replace(
                    env_state=runner_state.env_state.env_state.replace(
                        queued_state=dummy_queued_state,
                    )
                )
            )
            restored_ckpt['runner_state'] = runner_state

        # # Load the csv as a dataframe and delete all rows after the last checkpoint
        # progress_csv_path = os.path.join(get_exp_dir(config), "progress.csv")
        # progress_df = pd.read_csv(progress_csv_path, names=["timestep", "ep_return"])
        # # Convert timestep to int

        # progress_df = progress_df[progress_df["timestep"] <= steps_prev_complete]
        # progress_df.to_csv(progress_csv_path, header=False, index=False)

        return restored_ckpt

    if checkpoint_manager.latest_step() is None:
        restored_ckpt = None
    else:
        # print(f"Restoring checkpoint from {ckpt_dir}")
        # steps_prev_complete = checkpoint_manager.latest_step()

        ckpt_subdirs = os.listdir(ckpt_dir)
        ckpt_steps = [int(cs) for cs in ckpt_subdirs if cs.isdigit()]

        # Sort in decreasing order
        ckpt_steps.sort(reverse=True)
        for steps_prev_complete in ckpt_steps:
            try:
                restored_ckpt = try_load_ckpt(steps_prev_complete, target)
                if restored_ckpt is None:
                    raise TypeError("Restored checkpoint is None")
                break
            except TypeError as e:
                print(f"Failed to load checkpoint at step {steps_prev_complete}. Error: {e}")
                continue 
    
    return checkpoint_manager, restored_ckpt

    
def main_chunk(config, rng, exp_dir):
    """When jax jits the training loop, it pre-allocates an array with size equal to number of training steps. So, when training for a very long time, we sometimes need to break training up into multiple
    chunks to save on VRAM.
    """
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

    jax.block_until_ready(out)

    return out

    
@hydra.main(version_base=None, config_path='./conf', config_name='train_pcgrl')
def main(config: TrainConfig):
    config = init_config(config)
    if config.model == 'rnn':
        raise NotImplementedError("Single agent training script does not support `rnn` model.")
    rng = jax.random.PRNGKey(config.seed)

    exp_dir = config.exp_dir
    print(f'running experiment at {exp_dir}\n')


    # Need to do this before setting up checkpoint manager so that it doesn't refer to old checkpoints.
    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    if config.timestep_chunk_size != -1:
        n_chunks = config.total_timesteps // config.timestep_chunk_size
        for i in range(n_chunks):
            config.total_timesteps = config.timestep_chunk_size + (i * config.timestep_chunk_size)
            print(f"Running chunk {i+1}/{n_chunks}")
            out = main_chunk(config, rng, exp_dir)

    else:
        out = main_chunk(config, rng, exp_dir)        

#   ep_returns = out["runner_state"].ep_returns


if __name__ == "__main__":
    main()
