import copy
from dataclasses import asdict
import json
import os
import shutil

import hydra
import imageio
import jax
from jax import numpy as jnp
import numpy as np

from conf.config import GetTracesConfig
from envs.pcgrl_env import PCGRLEnv, PCGRLEnvState, render_stats, gen_dummy_queued_state
from envs.probs.problem import get_loss
from eval import get_eval_name, init_config_for_eval
from purejaxrl.experimental.s5.wrappers import LossLogEnvState, LossLogWrapper
from train import init_checkpointer
from utils import get_exp_dir, init_network, gymnax_pcgrl_make, init_config


@hydra.main(version_base="1.3", config_path='./conf', config_name='get_traces_pcgrl')
def main_get_traces(get_traces_config: GetTracesConfig):
    get_traces_config = init_config(get_traces_config)

    exp_dir = get_traces_config.exp_dir
    if get_traces_config.random_agent:
        # Save the gif of random agent behavior here. For debugging.
        os.makedirs(exp_dir)
        steps_prev_complete = 0
    else:
        if not os.path.exists(exp_dir):
            print(f"Experiment directory {exp_dir} does not exist")
            return
        print(f'Loading checkpoint from {exp_dir}')
        checkpoint_manager, restored_ckpt, wandb_run_id = init_checkpointer(get_traces_config)
        runner_state = restored_ckpt['runner_state']
        network_params = runner_state.train_state.params
        steps_prev_complete = restored_ckpt['steps_prev_complete']

    traces_dir = os.path.join(exp_dir, 'traces')

    if get_traces_config.overwrite_traces:
        print(f"Overwriting traces in {exp_dir}")
        if os.path.exists(traces_dir):
            print(f"Deleting traces directory {traces_dir}")
            shutil.rmtree(traces_dir)
    os.makedirs(traces_dir, exist_ok=True)

    env: PCGRLEnv

    # Preserve config as it was during training, for future reference (i.e. naming output of enjoy/eval)
    train_config = copy.deepcopy(get_traces_config)

    get_traces_config = init_config_for_eval(get_traces_config)
    env, env_params = gymnax_pcgrl_make(get_traces_config.env_name, config=get_traces_config)
    env = LossLogWrapper(env)
    env.prob.init_graphics()
    network = init_network(env, env_params, get_traces_config)

    rng = jax.random.PRNGKey(get_traces_config.eval_seed)
    rng_reset = jax.random.split(rng, get_traces_config.n_enjoy_envs)

    # Can manually define frozen tiles here, e.g. to set an OOD task
    # frz_map = jnp.zeros(env.map_shape, dtype=bool)
    # frz_map = frz_map.at[7, 3:-3].set(1)
    queued_state = gen_dummy_queued_state(env)
    # queued_state = env.queue_frz_map(queued_state, frz_map)

    # obs, env_state = env.reset(rng, env_params)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None, None))(
        rng_reset, env_params, queued_state
    )

    def step_env(carry, _):
        rng, obs, env_state = carry
        rng, rng_act = jax.random.split(rng)
        if get_traces_config.random_agent:
            action = env.action_space(env_params).sample(rng_act)[None, None, None, None]
        else:
            # obs = jax.tree_map(lambda x: x[None, ...], obs)
            action = network.apply(network_params, obs)[
                0].sample(seed=rng_act)
        rng_step = jax.random.split(rng, get_traces_config.n_enjoy_envs)
        # obs, env_state, reward, done, info = env.step(
        #     rng_step, env_state, action[..., 0], env_params
        # )
        obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            rng_step, env_state, action, env_params

        )
        frames = jax.vmap(env.render, in_axes=(0))(env_state.log_env_state.env_state)
        # frame = env.render(env_state)
        rng = jax.random.split(rng)[0]
        # Can't concretize these values inside jitted function (?)
        # So we add the stats on cpu later (below)
        # frame = render_stats(env, env_state, frame)
        return (rng, obs, env_state), (env_state, reward, done, info, frames)

    step_env = jax.jit(step_env)
    print('Scanning episode steps:')
    _, (states, rewards, dones, infos, frames) = jax.lax.scan(
        step_env, (rng, obs, env_state), None,
        length=get_traces_config.n_eps*env.max_steps)  # *at least* this many eps (maybe more if change percentage or whatnot)
    
    min_ep_losses = states.min_episode_losses
    # Mask out so we only have the final step of each episode
    min_ep_losses = jnp.where(dones, min_ep_losses, jnp.nan)

    # FIXME: get best frame index for *each* episode
    min_ep_loss_frame_idx = jnp.nanargmin(min_ep_losses, axis=0)

    # frames = frames.reshape((config.n_eps*env.max_steps, *frames.shape[2:]))

    # assert len(frames) == config.n_eps * env.max_steps, \
    #     "Not enough frames collected"
    assert frames.shape[1] == get_traces_config.n_enjoy_envs and frames.shape[0] == get_traces_config.n_eps * env.max_steps, \
        "`frames` has wrong shape"

    # Save gifs.
    print('Adding stats to json:')
    for env_idx in range(get_traces_config.n_enjoy_envs):
        # ep_frames = frames[ep_is*env.max_steps:(ep_is+1)*env.max_steps]

        for ep_idx in range(get_traces_config.n_eps):

            net_ep_idx = env_idx * get_traces_config.n_eps + ep_idx

            new_ep_frames = []

            for i in range(ep_idx * env.max_steps, (ep_idx + 1) * env.max_steps):
                # frame = frames[i, env_idx]
                # new_ep_frames.append(frame) # if we need image later
                
                state_i: LossLogEnvState = jax.tree_util.tree_map(lambda x: x[i, env_idx], states)
                # save all the stats of this current state to json
                env_state_i: PCGRLEnvState = state_i.log_env_state.env_state
                metric_name = env.prob.metric_names

                final_stats = {}

                final_stats["edit_position"] = env_state_i.rep_state.pos.tolist()
                final_stats["reward"] = env_state_i.reward.item()
                final_stats["pct_changed"] = env_state_i.pct_changed.item()
                final_stats["step_idx"] = env_state_i.step_idx.item()

                for hehe, s in enumerate(env_state_i.prob_state.stats):
                    final_stats[metric_name[hehe]] = {"current": s.item(), "target": env_state_i.prob_state.ctrl_trgs[hehe].item()}
                    
                vectorized_ascii = np.vectorize(lambda x: env.prob.tile_enum(x).ascii_char) 
                # without vectorize,This works fine for small maps but can be slow for large arrays since it's still using Python loops internally.
                # also this be applied element-wise on a NumPy array.
                ascii_map = vectorized_ascii(env_state_i.env_map)

                # or:
                # ascii_dict = {tile.value: tile.ascii_char for tile in env.prob.tile_enum}
                # ascii_map = np.vectorize(ascii_dict.get)(env_state_i.env_map)
                final_stats["map"] = ascii_map.tolist() # convert to list for json serialization

                # stats = env.prob.observe_ctrls(env_state_i.prob_state)
                # env_state_i = asdict(env_state_i)
                # env_state_i = convert_arrays(env_state_i)

                json_path = f"{traces_dir}/trace_{env_idx}-{net_ep_idx}/stats_ep-{net_ep_idx}_step-{i}.json"
               
                # print(f"Saving stats to {json_path}")
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                json.dump(final_stats, 
                          open(json_path, 'w'), 
                          indent=4) 
    print(f"Done saving traces to {traces_dir}")

    return 

def convert_arrays(obj):
    """ Recursively convert custom arrays to lists for JSON serialization """
    if isinstance(obj, dict):
        return {k: convert_arrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_arrays(v) for v in obj]
    elif hasattr(obj, "tolist"):  # Check if it's an array-like object
        return obj.tolist()
    else:
        return obj


if __name__ == '__main__':
    main_get_traces()
