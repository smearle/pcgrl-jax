import math
from typing import Optional, Tuple
import numpy as np
from flax import struct
from flax.core.frozen_dict import unfreeze
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
import chex

from envs.utils import Tiles


@struct.dataclass
class FloodPathState:
    flood_input: chex.Array
    flood_count: chex.Array
    env_map: Optional[chex.Array] = None
    trg: Optional[int] = None
    done: bool = False


@struct.dataclass
class FloodRegionsState:
    occupied_map: chex.Array
    flood_count: chex.Array
    done: bool = False


# FIXME: It's probably definitely (?) inefficient to use NNs here. Just use `jax.lax.convolve` directly.
#   (Also allows us to use ints instead of floats?)
class FloodPath(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(1, kernel_size=(3, 3), padding='SAME', kernel_init=constant(0.0), bias_init=constant(0.0))(x)
        return x

    def init_params(self, map_shape):
        rng = jax.random.PRNGKey(0) # This key doesn't matter since we'll reset before playing anyway(?)
        init_x = jnp.zeros(map_shape + (2,), dtype=jnp.float32)
        self.flood_params = unfreeze(self.init(rng, init_x))
        flood_kernel = self.flood_params['params']['Conv_0']['kernel']
        # Walls on center tile prevent it from being flooded
        flood_kernel = flood_kernel.at[1, 1, 0].set(-5)
        # Flood at adjacent tile produces flood toward center tile
        flood_kernel = flood_kernel.at[1, 2, 1].set(1)
        flood_kernel = flood_kernel.at[2, 1, 1].set(1)
        flood_kernel = flood_kernel.at[1, 0, 1].set(1) 
        flood_kernel = flood_kernel.at[0, 1, 1].set(1)
        flood_kernel = flood_kernel.at[1, 1, 1].set(1)
        self.flood_params['params']['Conv_0']['kernel'] = flood_kernel

    def flood_step(self, flood_state: FloodPathState):
        """Flood until no more tiles can be flooded."""
        flood_input, flood_count = flood_state.flood_input, flood_state.flood_count
        flood_params = self.flood_params
        occupied_map = flood_input[..., 0]
        flood_out = self.apply(flood_params, flood_input)
        flood_out = jnp.clip(flood_out, a_min=0, a_max=1)
        flood_out = jnp.stack([occupied_map, flood_out[..., -1]], axis=-1)
        flood_count = flood_out[..., -1] + flood_count
        done = jnp.all(flood_input == flood_out)
        flood_state = FloodPathState(flood_input=flood_out, flood_count=flood_count, done=done)
        return flood_state

    def flood_step_trg(self, flood_state: FloodPathState):
        """Flood until a target tile type is reached."""
        flood_input, flood_count = flood_state.flood_input, flood_state.flood_count
        trg = flood_state.trg
        flood_params = self.flood_params
        occupied_map = flood_input[..., 0]
        flood_out = self.apply(flood_params, flood_input)
        flood_out = jnp.clip(flood_out, a_min=0, a_max=1)
        flood_out = jnp.stack([occupied_map, flood_out[..., -1]], axis=-1)
        flood_count = flood_out[..., -1] + flood_count
        done = jnp.logical_or(
            jnp.all(flood_state.env_map != trg),
            jnp.logical_or(
                jnp.any(jnp.where(flood_state.env_map == trg, flood_count, False)),
                jnp.all(flood_input == flood_out),
            )
        )
        flood_state = FloodPathState(flood_input=flood_out, flood_count=flood_count, done=done,
                                     env_map=flood_state.env_map, trg=trg)
        return flood_state

    # def flood_step_while(self, flood_state: FloodPathState):
    #     flood_state, _ = self.flood_step(flood_state=flood_state)
    #     return flood_state

    # def flood_step_while_trg(self, flood_state: FloodPathState):
    #     flood_state, _ = self.flood_step_trg(flood_state=flood_state)
    #     return flood_state


class FloodRegions(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(5, kernel_size=(3, 3), padding='SAME', kernel_init=constant(0.0), bias_init=constant(0.0))(x)
        return x

    def init_params(self, map_shape):
        rng = jax.random.PRNGKey(0) # This key doesn't matter since we'll reset before playing anyway(?)
        init_x = jnp.zeros(map_shape + (1,), dtype=jnp.float32)
        self.flood_params = unfreeze(self.init(rng, init_x))
        flood_kernel = self.flood_params['params']['Conv_0']['kernel']
        flood_kernel = flood_kernel.at[1, 1, 0, 0].set(1)
        flood_kernel = flood_kernel.at[1, 2, 0, 1].set(1)
        flood_kernel = flood_kernel.at[2, 1, 0, 2].set(1)
        flood_kernel = flood_kernel.at[1, 0, 0, 3].set(1) 
        flood_kernel = flood_kernel.at[0, 1, 0, 4].set(1)
        self.flood_params['params']['Conv_0']['kernel'] = flood_kernel

    def flood_step(self, flood_regions_state: FloodRegionsState):
        """Flood until no more tiles can be flooded."""
        occupied_map, flood_count = flood_regions_state.occupied_map, flood_regions_state.flood_count
        flood_params = self.flood_params
        flood_out = self.apply(flood_params, flood_count)
        flood_count = jnp.max(flood_out, -1, keepdims=True)
        flood_count = flood_count * (1 - occupied_map[..., None])
        done = jnp.all(flood_count == flood_regions_state.flood_count)
        flood_regions_state = FloodRegionsState(flood_count=flood_count,
                                                occupied_map=occupied_map, done=done)
        return flood_regions_state

    # def flood_step(self, flood_state: FloodRegionsState, unused):
    #     done = flood_state.done
    #     # flood_state = jax.lax.cond(done, lambda _: flood_state, self._flood_step, flood_state)
    #     flood_state = self._flood_step(flood_state)
    #     return flood_state, None

    def flood_step_while(self, flood_state: FloodRegionsState):
        flood_state, _ = self.flood_step(flood_state=flood_state, unused=None)
        return flood_state

        
def get_path_coords(flood_count: chex.Array, max_path_len, coord1):
    y, x = coord1
    path_coords = jnp.full((max_path_len, 2), fill_value=-1, dtype=jnp.int32)
    path_coords = path_coords.at[0].set((y, x))
    curr_val = flood_count[y, x]
    max_val = jnp.max(flood_count)
    flood_count = jnp.where(flood_count == 0, jnp.inf, flood_count)
    padded_flood_count = jnp.pad(flood_count, 1, mode='constant', constant_values=jnp.inf)

    # while curr_val < max_val:
    def get_next_coord(carry):
        # Get the coordinates of a neighbor tile where the count is curr_val + 1
        curr_val, path_coords, padded_flood_count, i = carry
        last_yx = path_coords[i]
        padded_flood_count = padded_flood_count.at[last_yx[0]+1, last_yx[1]+1].set(jnp.inf)
        # nb_slice = padded_flood_count.at[last_xy[0]-1:last_xy[0] + 2, last_xy[1]-1:last_xy[1] + 2]
        nb_slice = jax.lax.dynamic_slice(padded_flood_count, (last_yx[0], last_yx[1]), (3, 3))
        # Mask out the corners to prevent diagonal movement
        nb_slice = nb_slice.at[[0, 0, 2, 2], [0, 2, 0, 2]].set(jnp.inf)
        # Get the coordinates of the minimum value
        y, x = jnp.argwhere(nb_slice==curr_val+1, size=1)[0]
        y, x = y - 1, x - 1
        yx = last_yx + jnp.array([y, x])
        path_coords = path_coords.at[i+1].set(yx)
        return curr_val+1, path_coords, padded_flood_count, i+1

    def cond(carry):
        curr_val, _, _, _ = carry
        return curr_val < max_val

    # path_coords = jax.lax.scan(get_next_coord, None, length=int(max_val - curr_val))
    _, path_coords, _, _ = jax.lax.while_loop(cond, get_next_coord, (curr_val, path_coords, padded_flood_count, 0))

    # while curr_val < max_val:
    #     get_next_coord()

    return path_coords


def get_path_coords_diam(flood_count: chex.Array, max_path_len):
    """Get the path coordinates from the flood count."""
    # Get the coordinates of a tile where the count is 1
    y, x = jnp.unravel_index(jnp.argmin(jnp.where(flood_count == 0, jnp.inf, flood_count)), flood_count.shape)
    return get_path_coords(flood_count, max_path_len, (y, x))


def get_max_path_length(map_shape: Tuple[int]):
    return math.ceil(math.prod(map_shape) / 2) + max(map_shape)


def get_max_n_regions(map_shape: Tuple[int]):
    return np.ceil(np.prod(map_shape) / 2).astype(int)  # Declaring this as static for jax.

    
def calc_n_regions(flood_regions_net: FloodRegions, env_map: chex.Array, passable_tiles: chex.Array):
    """Approximate the diameter of a maze-like tile map."""
    max_path_length = get_max_path_length(env_map.shape)

    # Get array of flattened indices of all tiles in env_map
    idxs = jnp.arange(math.prod(env_map.shape), dtype=jnp.float32) + 1
    regions_flood_count = idxs.reshape(env_map.shape)

    # Mask out flood_count where env_map is not empty
    occupied_map = (env_map[...,None] != passable_tiles).all(-1)
    # occupied_map = env_map != Tiles.EMPTY
    init_flood_count = regions_flood_count * (1 - occupied_map)

    flood_regions_state = FloodRegionsState(
            flood_count=init_flood_count[..., None], 
            occupied_map=occupied_map, done=False)
    # flood_regions_state, _ = jax.lax.scan(flood_regions_net.flood_step, flood_regions_state, jnp.arange(max_path_length))
    flood_regions_state = jax.lax.while_loop(
            lambda frs: jnp.logical_not(frs.done),
            flood_regions_net.flood_step,
            flood_regions_state)
    regions_flood_count = flood_regions_state.flood_count.astype(jnp.int32)

    # FIXME: Sketchily upper-bounding number of regions here since we need a concrete value
    n_regions = jnp.clip(jnp.unique(regions_flood_count, size=256, fill_value=0), 0, 1).sum()

    return n_regions


def calc_path_length(flood_path_net, env_map: jnp.ndarray, passable_tiles: jnp.ndarray, src: int, trg: int):
    occupied_map = (env_map[..., None] != passable_tiles).all(-1).astype(jnp.float32)
    init_flood = (env_map == src).astype(jnp.float32)
    init_flood_count = init_flood.copy()
    # Concatenate init_flood with new_occ_map
    flood_input = jnp.stack([occupied_map, init_flood], axis=-1)
    flood_state = FloodPathState(flood_input=flood_input, flood_count=init_flood_count, env_map=env_map, trg=trg)
    flood_state = jax.lax.while_loop(
            lambda fps: jnp.logical_not(fps.done),
            flood_path_net.flood_step_trg,
            flood_state)
    path_length = jnp.clip(flood_state.flood_count.max() - jnp.where(flood_state.flood_count == 0, 99999, flood_state.flood_count).min(), 0)
    return path_length, flood_state.flood_count


def calc_diameter(flood_regions_net: FloodRegions, flood_path_net: FloodPath, env_map: chex.Array, passable_tiles: chex.Array):
    """Approximate the diameter of a maze-like tile map."""
    max_path_length = get_max_path_length(env_map.shape)
    max_n_regions = get_max_n_regions(env_map.shape)

    # Get array of flattened indices of all tiles in env_map
    idxs = jnp.arange(math.prod(env_map.shape), dtype=jnp.float32) + 1
    regions_flood_count = idxs.reshape(env_map.shape)

    # Mask out flood_count where env_map is not empty
    occupied_map = (env_map[...,None] != passable_tiles).all(-1)
    # occupied_map = env_map != Tiles.EMPTY
    init_flood_count = regions_flood_count * (1 - occupied_map)

    # We'll use this for determining region anchors later
    pre_flood_count = (regions_flood_count + 1) * (1 - occupied_map)

    flood_regions_state = FloodRegionsState(flood_count=init_flood_count[..., None], occupied_map=occupied_map)
    # flood_regions_state, _ = jax.lax.scan(flood_regions_net.flood_step, flood_regions_state, jnp.arange(max_path_length))
    flood_regions_state = jax.lax.while_loop(
            lambda frs: jnp.logical_not(frs.done),
            flood_regions_net.flood_step,
            flood_regions_state)
    regions_flood_count = flood_regions_state.flood_count.astype(jnp.int32)

    # FIXME: Sketchily upper-bounding number of regions here since we need a concrete value
    n_regions = jnp.clip(jnp.unique(regions_flood_count, size=256, fill_value=0), 0, 1).sum()
    idxs = jnp.arange(math.prod(env_map.shape), dtype=jnp.float32)
    # path_length, flood_path_state = self.calc_path(env_map)
    regions_flood_count = flood_regions_state.flood_count[..., 0]

    # Because the maximum index value of each region has flooded throughout, we can select the tile at which this 
    # maximum index value originally appeared as the "anchor" of this region.
    region_anchors = jnp.clip(pre_flood_count - regions_flood_count, 0, 1)

    # Now we flood out from all of these anchor points and find the furthest point from each.
    init_flood = region_anchors
    init_flood_count = init_flood
    flood_input = jnp.stack([occupied_map, init_flood], axis=-1)
    flood_path_state = FloodPathState(flood_input=flood_input, flood_count=init_flood_count)
    # flood_path_state, _ = jax.lax.scan(flood_path_net.flood_step, flood_path_state, None, max_path_length)
    flood_path_state = jax.lax.while_loop(
            lambda state: jnp.logical_not(state.done),
            flood_path_net.flood_step,
            flood_path_state)

    # We need to find the max path length in *each region*. So we'll mask out the path lengths of all other regions.
    # Unique (max) region indices
    region_idxs = jnp.unique(regions_flood_count, size=max_n_regions, fill_value=0)
    region_masks = jnp.where(regions_flood_count[..., None] == region_idxs, 1, 0)
    path_flood_count = flood_path_state.flood_count
    region_path_floods = path_flood_count[..., None] * region_masks
    region_path_floods = region_path_floods.reshape((-1, region_path_floods.shape[-1]))
    # Now we identify the furthest point from the anchor in each region.
    region_endpoint_idxs = jnp.argmin(jnp.where(region_path_floods == 0, max_path_length * 2, region_path_floods), axis=0)
    region_endpoint_idxs = jnp.unravel_index(region_endpoint_idxs, env_map.shape)
    init_flood = jnp.zeros(env_map.shape)
    # We can now flood out from these endpoints.
    init_count = init_flood = init_flood.at[region_endpoint_idxs].set(1) * (1 - occupied_map)
    flood_input = jnp.stack([occupied_map, init_flood], axis=-1)
    flood_path_state = FloodPathState(flood_input=flood_input, flood_count=init_count)
    # flood_path_state, _ = jax.lax.scan(flood_path_net.flood_step, flood_path_state, None, max_path_length)
    flood_path_state = jax.lax.while_loop(
            lambda state: jnp.logical_not(state.done),
            flood_path_net.flood_step,
            flood_path_state)
    path_length = jnp.clip(flood_path_state.flood_count.max() - jnp.where(flood_path_state.flood_count == 0, max_path_length, flood_path_state.flood_count).min(), 0)

    return path_length, flood_path_state, n_regions, flood_regions_state
