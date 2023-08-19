import os
import shutil
import hydra
import jax
import jax.numpy as jnp
from flax import struct
import optax
from typing import NamedTuple
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax

from config import Config, TrainConfig
from purejaxrl.experimental.s5.wrappers import LogWrapper
from utils import (get_ckpt_dir, get_exp_dir, get_network, gymnax_pcgrl_make,
                   init_config)


class RunnerState(struct.PyTreeNode):
    train_state: TrainState
    env_state: jnp.ndarray
    last_obs: jnp.ndarray
    rng_act: jnp.ndarray
    ep_returns: jnp.ndarray
    rng: jnp.ndarray


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    rng_act: jnp.ndarray


def make_train(config: TrainConfig, restored_ckpt, checkpoint_manager):
    config.NUM_UPDATES = (
        config.total_timesteps // config.num_steps // config.num_envs
    )
    config.MINIBATCH_SIZE = (
        config.num_envs * config.num_steps // config.NUM_MINIBATCHES
    )
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    # env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.NUM_MINIBATCHES * config.update_epochs))
            / config.NUM_UPDATES
        )
        return config["LR"] * frac

    def train(rng, config: TrainConfig):
        # INIT NETWORK
        network = get_network(env, env_params, config)

        rng, _rng = jax.random.split(rng)
        rng_act = jax.random.split(_rng, config.num_envs)
        init_x = jnp.zeros((1,) + env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        print(network.subnet.tabulate(_rng, init_x))

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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        rng_act = reset_rng
        obsv, env_state = jax.vmap(
            env.reset, in_axes=(0, None))(reset_rng, env_params)

        def save_checkpoint(runner_state, info, steps_prev_complete):
            try:
                timesteps = info["timestep"][info["returned_episode"]
                                             ] * config.num_envs
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

        rng, _rng = jax.random.split(rng)
        ep_returns = jnp.full(shape=config.NUM_UPDATES,
                              fill_value=jnp.nan, dtype=jnp.float32)
        steps_prev_complete = 0
        runner_state = RunnerState(
            train_state, env_state, obsv, rng_act, ep_returns, _rng)

        # exp_dir = get_exp_dir(config)
        if restored_ckpt is not None:
            # target = {'runner_state': runner_state, 'config': config, 'step_i': 0}
            # raw_restored = checkpoint_manager.restore(checkpoint_manager.latest_step(), items=target)
            # raw_restored = checkpoint_manager.restore(checkpoint_manager.latest_step())
            # runner_state = raw_restored['runner_state']
            # config = raw_restored['config']
            steps_prev_complete = restored_ckpt['steps_prev_complete']
            runner_state = restored_ckpt['runner_state']
            steps_remaining = config.total_timesteps - steps_prev_complete
            config.NUM_UPDATES = int(
                steps_remaining // config.num_steps // config.num_envs)

            # TODO: Overwrite certain config values

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state: RunnerState, unused):
                train_state, env_state, last_obs, ep_returns, rng = (
                    runner_state.train_state, runner_state.env_state,
                    runner_state.last_obs, runner_state.ep_returns,
                    runner_state.rng
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                rng_act = jax.random.split(_rng, config.num_envs)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action[..., None], env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info,
                    rng_act
                )
                runner_state = RunnerState(
                    train_state, env_state, obsv, rng_act, ep_returns, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng_act, ep_returns, rng = runner_state.train_state, runner_state.env_state, \
                runner_state.last_obs, runner_state.rng_act, runner_state.ep_returns, runner_state.rng
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
                        pi, value = network.apply(params, traj_batch.obs)
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
                        # if config.representation == 'nca':
                        gae = gae[..., None, None]
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
                    batch_size == config.num_steps * config.num_envs
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
                                                 ] * config.num_envs
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, metric, steps_prev_complete)

            jax.debug.callback(save_checkpoint, runner_state,
                               metric, steps_prev_complete)

            def log_callback(metric, steps_prev_complete, ep_returns):
                timesteps = metric["timestep"][metric["returned_episode"]
                                               ] * config.num_envs
                if len(timesteps) > 0:
                    t = timesteps[0]
                    ep_return = metric["returned_episode_returns"][metric["returned_episode"]].mean(
                    )
                    # ep_returns = ep_returns.at[t].set(ep_return)

                    # Add a row to csv with ep_return
                    with open(os.path.join(get_exp_dir(config), "progress.csv"), "a") as f:
                        f.write(f"{t},{ep_return}\n")

            jax.debug.callback(log_callback, metric,
                               steps_prev_complete, ep_returns)

            runner_state = RunnerState(
                train_state, env_state, last_obs, rng_act, ep_returns, rng)

            return runner_state, metric

        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config.NUM_UPDATES
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
    rng = jax.random.PRNGKey(30)
    # Set up checkpointing
    ckpt_dir = get_ckpt_dir(config)

    # Create a dummy checkpoint so we can restore it to the correct dataclasses
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    # env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    rng, _rng = jax.random.split(rng)
    rng_act = jax.random.split(_rng, config.num_envs)
    network = get_network(env, env_params, config)
    init_x = jnp.zeros((1,) + env.observation_space(env_params).shape)
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
    reset_rng = jax.random.split(_rng, config.num_envs)
    rng_act = reset_rng
    obsv, env_state = jax.vmap(
        env.reset, in_axes=(0, None))(reset_rng, env_params)
    runner_state = RunnerState(train_state=train_state, env_state=env_state, last_obs=obsv, rng_act=rng_act,
                               ep_returns=jnp.full(config.num_envs, jnp.nan), rng=rng)
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


@hydra.main(version_base=None, config_path='./', config_name='train')
def main(config: TrainConfig):
    config = init_config(config)
    rng = jax.random.PRNGKey(config.seed)

    exp_dir = get_exp_dir(config)

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
            "from. How can this be?"
        # Create csv for logging progress
        with open(os.path.join(exp_dir, "progress.csv"), "w") as f:
            f.write("timestep,ep_return\n")

    train_jit = jax.jit(make_train(config, restored_ckpt, checkpoint_manager))
    out = train_jit(rng)

    ep_returns = out["runner_state"].ep_returns


if __name__ == "__main__":
    main()
