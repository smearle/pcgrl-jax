
from abc import ABC
from typing import Optional
import chex
from flax import struct
import jax.numpy as jnp

from gymnax.environments import spaces


@struct.dataclass
class RepresentationState:
    pass


class Representation(ABC):
    def __init__(self, tile_enum):
        self.tile_enum = tile_enum

    def observation_shape(self):
        return (*self.rf_shape, len(self.tile_enum))

    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        observation_shape = self.observation_shape()
        low = jnp.zeros(
            observation_shape,
            # self.rf_size ** 2 * self.num_tiles * [0],
            # self.rf_size ** 2 * self.num_tiles * [0] + self.num_actions * [0] + [0, 0],
            dtype=jnp.float32,
        )
        high = jnp.zeros(
            observation_shape,
            # self.rf_size ** 2 * self.num_tiles * [1],
            # + self.num_actions * [1]
            # + [1, params.max_steps_in_episode],
            dtype=jnp.float32,
        )
        return spaces.Box(
            low, high, observation_shape, jnp.float32
            # low, high, (self.rf_size ** 2 * self.num_tiles + self.num_actions + 2,), jnp.float32
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.tile_enum) - 1)

    def get_obs(self) -> chex.Array:
        raise NotImplementedError