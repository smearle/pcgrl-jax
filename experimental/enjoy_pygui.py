import PySimpleGUI as sg
import hydra
import imageio
import os
import jax
from jax import numpy as jnp
import numpy as np

# Assuming all the necessary functions and classes are defined as in the original script
from conf.config import EnjoyConfig
from envs.pcgrl_env import PCGRLEnv, render_stats, gen_dummy_queued_state
from train import init_checkpointer
from utils import get_exp_dir, init_network, gymnax_pcgrl_make, init_config


@hydra.main(version_base="1.3", config_path='./', config_name='enjoy_pcgrl')
def enjoy_pygui(config: EnjoyConfig):
    init_config(config)
    exp_dir = config.exp_dir

    print(f'Loading checkpoint from {exp_dir}')
    checkpoint_manager, restored_ckpt = init_checkpointer(config)
    runner_state = restored_ckpt['runner_state']
    network_params = runner_state.train_state.params
    steps_prev_complete = restored_ckpt['steps_prev_complete']


    # Initialize the GUI
    layout = [
        [sg.Text('RL Agent Viewer')],
        [sg.Image(filename='', key='image')],
        [sg.Button('Start'), sg.Button('Pause'), sg.Button('Reset'), sg.Button('Exit')]
    ]

    window = sg.Window('RL Agent Simulation', layout)

    # Global state for simulation control
    simulation_running = False
    paused = False

    # Initialize environment and agent (simplified)
    # Simplified initialization, assuming configuration is handled separately
    config = init_config(config)
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    network = init_network(env, env_params, config)
    rng = jax.random.PRNGKey(0)

    # Main event loop
    while True:
        event, values = window.read(timeout=100)  # Poll every 100 ms

        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Start':
            simulation_running = True
            paused = False
            # Reset the environment and agent
            obs, env_state = env.reset()
        elif event == 'Pause':
            paused = not paused  # Toggle pause state
        elif event == 'Reset':
            simulation_running = True
            paused = False
            # obs, env_state = env.reset(...)

        if simulation_running and not paused:


            def step_env(carry, _):
                rng, obs, env_state = carry
                rng, rng_act = jax.random.split(rng)
                if config.random_agent:
                    action = env.action_space(env_params).sample(rng_act)
                else:
                    # obs = jax.tree_map(lambda x: x[None, ...], obs)
                    action = network.apply(network_params, obs)[
                        0].sample(seed=rng_act)
                rng_step = jax.random.split(rng, config.n_eps)
                # obs, env_state, reward, done, info = env.step(
                #     rng_step, env_state, action[..., 0], env_params
                # )
                obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params

                )
                frames = jax.vmap(env.render, in_axes=(0))(env_state)
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
                length=1)
            # Update the environment with the agent's action and render
            # This part should be adapted from the script's logic
            # For example:
            # action = network.apply(...)  # Get the agent's action
            # obs, env_state, reward, done, info = env.step(action)
            # frame = env.render(env_state)  # Get the current frame to display
            # Update GUI with new frame
            # window['image'].update(data=frame)  # Assuming frame is in a format PySimpleGUI can display

    window.close()
