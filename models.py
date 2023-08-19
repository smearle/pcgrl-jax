import math
from timeit import default_timer as timer
from typing import Sequence, Tuple

import chex
import distrax
from flax.linen.initializers import constant, orthogonal
import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp


class Dense(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, _):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        act = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        act = activation(act)
        act = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(act)
        act = activation(act)
        act = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class ConvForward(nn.Module):
    action_dim: Sequence[int]
    arf_size: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Slice out a square of width `arf_size` from `x`
        mid_x = x.shape[2] // 2
        mid_y = x.shape[3] // 2
        act = x[:, :,
                mid_x-math.floor(self.arf_size/2):
                mid_x+math.ceil(self.arf_size/2),
                mid_y-math.floor(self.arf_size/2):
                mid_y+math.ceil(self.arf_size/2)]

        act = nn.Conv(
            features=64, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(act)
        act = activation(act)
        act = nn.Conv(
            features=64, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(act)
        act = activation(act)
        act = act.reshape((x.shape[0], -1))
        act = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(act)
        act = activation(act)
        act = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        critic = x.reshape((x.shape[0], -1))
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class SeqNCA(nn.Module):
    action_dim: Sequence[int]
    arf_size: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        hid = nn.Conv(
            features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME"
        )(x)
        hid = activation(hid)

        # Slice out a square of width `arf_size` from `x`
        mid_x = hid.shape[2] // 2
        mid_y = hid.shape[3] // 2
        act = hid[:, :,
                  mid_x-math.floor(self.arf_size/2):
                      mid_x+math.ceil(self.arf_size/2),
                  mid_y-math.floor(self.arf_size/2):
                      mid_y+math.ceil(self.arf_size/2)]
        act = act.reshape((x.shape[0], -1))
        act = nn.Dense(
            64, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(act)
        act = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        critic = hid.reshape((x.shape[0], -1))
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class NCA(nn.Module):
    representation: str
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        act = nn.Conv(features=256, kernel_size=(9, 9), padding="SAME")(x)
        act = activation(act)
        act = nn.Conv(features=256, kernel_size=(5, 5), padding="SAME")(act)
        act = activation(act)
        act = nn.Conv(features=self.action_dim,
                      kernel_size=(3, 3), padding="SAME")(act)

        if self.representation == 'wide':
            act = act.reshape((x.shape[0], -1))

        # Generate random binary mask
        # mask = jax.random.uniform(rng[0], shape=actor_mean.shape) > 0.9
        # Apply mask to logits
        # actor_mean = actor_mean * mask
        # actor_mean = (actor_mean + x) / 2

        # actor_mean *= 10
        # actor_mean = nn.softmax(actor_mean, axis=-1)

        # critic = nn.Conv(features=256, kernel_size=(3,3), padding="SAME")(x)
        # critic = activation(critic)
        # # actor_mean = nn.Conv(
        #       features=256, kernel_size=(3,3), padding="SAME")(actor_mean)
        # # actor_mean = activation(actor_mean)
        # critic = nn.Conv(
        #       features=1, kernel_size=(1,1), padding="SAME")(critic)

        # return act, critic

        critic = x.reshape((x.shape[0], -1))
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class AutoEncoder(nn.Module):
    representation: str
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        act = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2),
                      padding="SAME")(x)
        act = activation(act)
        act = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2),
                      padding="SAME")(act)
        act = activation(act)
        act = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2),
                               padding="SAME")(act)
        act = activation(act)
        act = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2),
                               padding="SAME")(act)
        act = activation(act)
        act = nn.Conv(features=self.action_dim,
                      kernel_size=(3, 3), padding="SAME")(act)

        if self.representation == 'wide':
            act = act.reshape((x.shape[0], -1))

        critic = x.reshape((x.shape[0], -1))
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class ActorCritic(nn.Module):
    """Transform the action output into a distribution."""
    subnet: nn.Module
    act_shape: Tuple[int, int]

    @nn.compact
    def __call__(self, x: chex.Array):
        act, val = self.subnet(x)
        act = act.reshape((x.shape[0], *self.act_shape, -1))
        pi = distrax.Categorical(logits=act)

        return pi, val


if __name__ == '__main__':
    n_trials = 100
    rng = jax.random.PRNGKey(42)
    start_time = timer()
    for _ in range(n_trials):
        rng, _rng = jax.random.split(rng)
        data = jax.random.normal(rng, (4, 256, 2))
        print('data', data)
        dist = distrax.Categorical(data)
        sample = dist.sample(seed=rng)
        print('sample', sample)
        log_prob = dist.log_prob(sample)
        print('log_prob', log_prob)
    time = timer() - start_time
    print(f'Average time per sample: {time / n_trials}')
