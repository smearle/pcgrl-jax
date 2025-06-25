import os
from typing import Sequence

import imageio

from envs.pcgrl_env import PCGRLEnv, PCGRLEnvParams, PCGRLEnvState, PCGRLObs
from flax import linen as nn
import hydra
import jax
from jax import numpy as jnp

from conf.config import Config
from utils import gymnax_pcgrl_make, init_config


class NCA(nn.Module):
    tile_action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = nn.Conv(features=256, kernel_size=(9, 9), padding="SAME")(x)
        x = activation(x)
        x = nn.Conv(features=256, kernel_size=(5, 5), padding="SAME")(x)
        x = activation(x)
        x = nn.Conv(features=self.tile_action_dim,
                      kernel_size=(3, 3), padding="SAME")(x)

        return x


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Config):
    init_config(cfg)
    env, env_params = gymnax_pcgrl_make("PCGRL", config=cfg)
    env: PCGRLEnv
    env_params: PCGRLEnvParams
    env.init_graphics()
    key = jax.random.PRNGKey(0)

    _, env_state = env.reset(key, env_params)
    env_state: PCGRLEnvState
    env_map = env_state.env_map

    def step_nca(carry, _):
        env_map = carry
        n_tiles = env.rep.tile_action_dim
        nca = NCA(tile_action_dim=env.rep.tile_action_dim)
        x = env_map
        x = jax.nn.one_hot(x, n_tiles + 1)
        network_params = nca.init(key, x=x)
        out = nca.apply(network_params, x)
        env_map = out.argmax(axis=-1) + 1
        return env_map, env_map

    env_map, env_maps = jax.lax.scan(step_nca, env_map, length=20)

    prob_states = jax.vmap(env.prob.get_curr_stats)(env_maps)
    env_states = jax.vmap(
        lambda env_state, env_map, prob_state: env_state.replace(
            env_map=env_map,
            prob_state=prob_state,
        ),
        in_axes=(None, 0, 0)
    )(
        env_state, env_maps, prob_states,
    )

    frames = jax.vmap(env.render)(env_states)

    for i, frame in enumerate(frames):
        im_path = (os.path.join('temp', f'im-{i}.png'))
        imageio.imsave(im_path, frame)


if __name__ == '__main__':
    main()