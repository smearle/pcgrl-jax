from abc import ABC
import math
from typing import Tuple

import chex
from flax import struct
from gymnax.environments import spaces
import jax
import jax.numpy as jnp

from envs.utils import Tiles


@struct.dataclass
class RepresentationState:
    pass


class Representation(ABC):
    def __init__(self, tile_enum: Tiles, rf_shape: Tuple[int, int],
                 act_shape: Tuple[int, int]):
        self.tile_enum = tile_enum
        self.act_shape = act_shape

    def observation_shape(self):
        # Always observe static tile channel
        return (*self.rf_shape, len(self.tile_enum) + 1)

    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        observation_shape = self.observation_shape()
        low = 1
        high = 1
#       low = jnp.zeros(
#           observation_shape,
#           dtype=jnp.float32,
#       )
#       high = jnp.zeros(
#           observation_shape,
#           dtype=jnp.float32,
#       )
        return spaces.Box(
            low, high, observation_shape, jnp.float32
        )

    def action_space(self) -> spaces.Discrete:
        # return spaces.Discrete(len(self.tile_enum) - 1)
        return spaces.Discrete((len(self.tile_enum)-1)
                               * math.prod(self.act_shape))

    def get_obs(self) -> chex.Array:
        raise NotImplementedError


def get_ego_obs(self, env_map: chex.Array, static_map: chex.Array,
                rep_state: RepresentationState):
    padded_env_map = jnp.pad(
        env_map, self.rf_off, mode='constant',
        constant_values=self.tile_enum.BORDER)
    rf_map_obs = jax.lax.dynamic_slice(
        padded_env_map,
        rep_state.pos,
        self.rf_shape,
    )
    # Convert to one-hot encoding
    rf_obs = jax.nn.one_hot(rf_map_obs, self.num_tiles)
    if static_map is not None:
        padded_static_map = jnp.pad(static_map, self.rf_off, mode='constant',
                                    constant_values=1)  # Border is static
        rf_static_obs = jax.lax.dynamic_slice(
            padded_static_map,
            rep_state.pos,
            self.rf_shape,
        )
        rf_obs = jnp.concatenate([rf_obs, rf_static_obs[..., None]], axis=-1)
    return rf_obs


def get_global_obs(self, env_map: chex.Array, static_map: chex.Array,
                   rep_state: RepresentationState):
    # Convert to one-hot encoding
    rf_obs = jax.nn.one_hot(env_map - 1, self.num_tiles - 1)
    if static_map is not None:
        rf_static_obs = jnp.expand_dims(static_map, axis=-1)
        rf_obs = jnp.concatenate([rf_obs, rf_static_obs], axis=-1)
    return rf_obs
