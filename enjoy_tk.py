import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import hydra
import imageio
import jax
from jax import numpy as jnp
import numpy as np

from conf.config import EnjoyConfig
from envs.pcgrl_env import PCGRLEnv, PCGRLEnvState, PCGRLObs, render_stats
from train import init_checkpointer
from utils import get_exp_dir, init_network, gymnax_pcgrl_make, init_config


def set_ctrl_trgs(env, trgs, env_state: PCGRLEnvState):
    old_trgs = env_state.prob_state.ctrl_trgs
    new_trgs = old_trgs
    for k, v in trgs.items():
        metric_i = env.prob.metrics_enum[k]
        new_trgs = new_trgs.at[metric_i].set(v)
    env_state = env_state.replace(prob_state=env_state.prob_state.replace(ctrl_trgs=new_trgs))
    return env_state


@hydra.main(version_base=None, config_path='./', config_name='enjoy_pcgrl')
def main_enjoy_tk(config: EnjoyConfig):
    config = init_config(config)

    exp_dir = config.exp_dir
    if not config.random_agent:
        checkpoint_manager, restored_ckpt = init_checkpointer(config)
        network_params = restored_ckpt['runner_state'].train_state.params
    elif not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    env: PCGRLEnv
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env.prob.init_graphics()
    network = init_network(env, env_params, config)

    rng = jax.random.PRNGKey(42)
    rng_reset, _ = jax.random.split(rng)

    obs: PCGRLObs
    env_state: PCGRLEnvState
    obs, env_state = env.reset(rng, env_params)

    metric_names = env.prob.metric_names
    metric_bounds = env.prob.metric_bounds
    metric_trgs = env_state.prob_state.ctrl_trgs
    metric_vals = env_state.prob_state.stats

    root = tk.Tk()
    root.title("PCGRL Enjoy")

    # Image Label
    img_label = tk.Label(root)
    img_label.pack()

    # Pause Button
    paused = tk.BooleanVar(value=False)
    def toggle_pause():
        paused.set(not paused.get())
    pause_button = tk.Button(root, text="Pause/Resume", command=toggle_pause)
    pause_button.pack()

    # Reset Button
    def reset_episode():
        nonlocal obs, env_state
        obs, env_state = env.reset(rng_reset, env_params)
    reset_button = tk.Button(root, text="Reset", command=reset_episode)
    reset_button.pack()

    # Metrics Progress Bars and Control Target Scales
    progress_bars = {}
    scales = {}
    for metric in metric_names:
        # Get metric enum
        metric_i = env.prob.metrics_enum[metric]
        bounds = metric_bounds[metric_i]
        ttk.Label(root, text=metric).pack()
        progress = ttk.Progressbar(root, maximum=bounds[1], value=0)
        progress.pack()
        progress_bars[metric] = progress

        scale = tk.Scale(root, from_=bounds[0], to=bounds[1], orient="horizontal")
        scale.set(metric_trgs[metric_i])
        scale.pack()
        scales[metric] = scale

    def step_env(carry, _):
        rng, obs, env_state = carry
        rng, rng_act = jax.random.split(rng)
        if config.random_agent:
            action = env.action_space(env_params).sample(rng_act)
        else:
            obs = jax.tree_map(lambda x: x[None, ...], obs)
            action = network.apply(network_params, obs)[
                0].sample(seed=rng_act)
        rng_step, _ = jax.random.split(rng)
        env_state: PCGRLEnvState
        obs, env_state, reward, done, info = env.step(
            rng_step, env_state, action[..., 0], env_params
        )
        metric_vals = env_state.prob_state.stats
        frame = env.render(env_state)

        # Tkinter UI updates
        img = Image.fromarray(np.array(frame))  # Assume frame is compatible
        img = ImageTk.PhotoImage(image=img)
        img_label.config(image=img)
        img_label.image = img

        for metric_i, value in enumerate(metric_vals):
            metric = metric_names[metric_i]
            progress_bars[metric].config(value=value)
        
        for metric, scale in scales.items():
            env_state = set_ctrl_trgs(env, {metric: scale.get()}, env_state)

        return (rng, obs, env_state), (env_state, reward, done, info, frame)

    def game_loop():
        nonlocal rng, obs, env_state
        if not paused.get():
            _, (env_state, reward, done, info, frame) = step_env((rng, obs, env_state), None)
        root.after(1, game_loop)

    game_loop()
    root.mainloop()


if __name__ == '__main__':
    main_enjoy_tk()
