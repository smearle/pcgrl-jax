import numpy as np
import hydra
import jax
import pygame
from conf.config import EnjoyConfig
from envs.pcgrl_env import PCGRLEnvState, render_stats
from train import init_checkpointer
from utils import gymnax_pcgrl_make, init_config, init_network

scale = 3

def init_pygame(env):
    pygame.init()
    screen_width, screen_height = 800 * scale, 600 * scale
    screen = pygame.display.set_mode((screen_width, screen_height))
    return screen

def np_array_to_surface(array):
    array = array.astype(np.uint8)
    array = array.swapaxes(0, 1)
    return pygame.surfarray.make_surface(array)

@hydra.main(version_base="1.3", config_path='./', config_name='enjoy_pcgrl')
def enjoy_pygame(config: EnjoyConfig):
    config = init_config(config)
    if not config.random_agent:
        ckpt_manager, restored_ckpt = init_checkpointer(config)
    rng = jax.random.PRNGKey(000)
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env.init_graphics()
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    freeze_builds = False

    network = init_network(env, env_params, config)
    if not config.random_agent:
        network_params = restored_ckpt['runner_state'].train_state.params
    else:
        pass
        # init_obs = jax.tree_map(lambda x: x[None], obs)
        # network_params = network.init(rng, init_obs)
        # network_params = None

    screen = init_pygame(env)
    paused = False  # Initial state of the game is not paused

    user_selected_tile = 0

    env_state: PCGRLEnvState

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:  # Pause toggle
                    paused = not paused
                elif event.key == pygame.K_r:  # Reset
                    rng, rng_reset = jax.random.split(rng)
                    obs, env_state = env.reset(rng_reset, env_params)
                elif event.key == pygame.K_f:  # Freeze builds toggle
                    freeze_builds = not freeze_builds 
                
                # If the key is 0-9, set the corresponding tile as being selected
                elif event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]: 
                    # Convert the key to an integer (0-9)
                    tile = int(event.unicode)
                    user_selected_tile = tile
                    print(tile)

            # If the user has clicked their mouse
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get the position of the mouse
                pos = pygame.mouse.get_pos()
                print(pos)
                # Get coordinates in the env_map
                x = pos[0] // (scale * env.tile_size) - 1
                y = pos[1] // (scale * env.tile_size) - 1
                print(x, y)

                env_map = env_state.env_map
                env_map = env_map.at[y, x].set(user_selected_tile)
                print(f'shape: {env_map.shape}')

                env_state = env_state.replace(env_map=env_map)

                env_state = env_state.replace(static_map=env_state.static_map.at[y, x].set(int(freeze_builds)))


        if not paused:
            obs = jax.tree_map(lambda x: x[None], obs)
            if config.random_agent:
                action = env.action_space(env_params).sample(rng)
                action = jax.tree_map(lambda x: x[None], action)
            else:
                action = network.apply(network_params, obs)[0].sample(seed=rng)
                action = action[0]
            obs, env_state, reward, done, info = env.step(rng, env_state, action, env_params)

        frame = np.array(env.render(env_state))[:, :, :3]
        frame = np.kron(frame, np.ones((scale, scale, 1)))
        frame_surface = np_array_to_surface(frame)
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        pygame.time.wait(0)

if __name__ == '__main__':
    enjoy_pygame()
