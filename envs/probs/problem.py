from enum import IntEnum
from functools import partial
import math
from typing import Optional

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.utils import Tiles


class Stats(IntEnum):
    pass


@struct.dataclass
class ProblemState:
    stats: Optional[chex.Array] = None

    # FIXME: A bit weird how we handle this, setting as None all over the place in problem classes...
    ctrl_trgs: Optional[chex.Array] = None


def get_reward(stats, old_stats, stat_weights, stat_trgs, ctrl_threshes):
    """
    ctrl_threshes: A vector of thresholds for each metric. If the metric is within
        an interval of this size centered at its target value, it has 0 loss.
    """
    if stat_trgs.ndim > 1: # for multi-agent setting
        stats = stats[..., None]
        old_stats = old_stats[..., None]
        ctrl_threshes = ctrl_threshes[..., None]
    prev_loss = jnp.abs(stat_trgs - old_stats)
    prev_loss = jnp.clip(prev_loss - ctrl_threshes, 0)    
    loss = jnp.abs(stat_trgs - stats)
    loss = jnp.clip(loss - ctrl_threshes, 0)
    reward = prev_loss - loss
    reward = jnp.where(stat_trgs == jnp.inf, stats - old_stats, reward)
    reward = jnp.where(stat_trgs == -jnp.inf, old_stats - stats, reward)
    reward *= stat_weights
    if stat_trgs.ndim > 1:
        reward = jnp.sum(reward, axis=0)
    else:
        reward = jnp.sum(reward)
    return reward

    
def get_max_loss(stat_weights, stat_trgs, ctrl_threshes, metric_bounds):
    if stat_trgs.ndim > 1:
        metric_bounds = metric_bounds[..., None]
        ctrl_threshes = ctrl_threshes[..., None]
    stat_trgs = jnp.clip(stat_trgs, metric_bounds[:, 0], metric_bounds[:, 1])
    loss_0 = jnp.abs(stat_trgs - metric_bounds[:, 0])
    loss_1 = jnp.abs(stat_trgs - metric_bounds[:, 1])
    loss = jnp.where(loss_0 < loss_1, loss_1, loss_0)
    loss = jnp.clip(loss - ctrl_threshes, 0)
    loss *= stat_weights
    loss = jnp.sum(loss)
    return loss

    
def get_loss(stats, stat_weights, stat_trgs, ctrl_threshes, metric_bounds):
    stat_trgs = jnp.clip(stat_trgs, metric_bounds[:, 0], metric_bounds[:, 1])
    loss = jnp.abs(stat_trgs - stats)
    loss = jnp.clip(loss - ctrl_threshes, 0)
    loss *= stat_weights
    loss = jnp.sum(loss)
    return loss


def gen_ctrl_trgs(metric_bounds, rng):
    rng, _ = jax.random.split(rng)
    return jax.random.randint(rng, (len(metric_bounds),), metric_bounds[:, 0], metric_bounds[:, 1])

    
@struct.dataclass
class MapData:
    env_map: chex.Array
    actual_map_shape: chex.Array


@partial(jax.jit, static_argnames=("tile_enum", "map_shape", "tile_probs", "randomize_map_shape", "empty_start", 
                                   "tile_nums", "pinpoints"))
def gen_init_map(rng, tile_enum, map_shape, tile_probs, randomize_map_shape=False, empty_start=False, tile_nums=None,
                 pinpoints=False):
    tile_probs = np.array(tile_probs, dtype=np.float32)

    if empty_start:
        init_map = jnp.full(map_shape, dtype=jnp.int32, fill_value=tile_enum.EMPTY)
    else:
        # Randomly place tiles according to their probabilities tile_probs
        init_map = jax.random.choice(rng, len(tile_enum), shape=map_shape, p=tile_probs)

    if randomize_map_shape:
        # Randomize the actual map size
        actual_map_shape = jax.random.randint(rng, (2,), 3, jnp.max(jnp.array(map_shape)) + 1)

        # Use jnp.ogrid to create a grid of indices
        oy, ox = jnp.ogrid[:map_shape[0], :map_shape[1]]
        # Use these indices to create a mask where each dimension is less than the corresponding actual_map_shape
        mask = (oy < actual_map_shape[0]) & (ox < actual_map_shape[1])

        # Replace the rest with tile_enum.BORDER
        init_map = jnp.where(mask, init_map, tile_enum.BORDER)
    
    else:
        actual_map_shape = jnp.array(map_shape)

    if tile_nums is not None and pinpoints:

        non_num_tiles = jnp.array([tile_idx for tile_idx, tile_num in enumerate(tile_nums) if tile_num == 0])
        n_map_cells = math.prod(map_shape)

        def add_num_tiles(carry, tile_idx):
            rng, init_map = carry
            tiles_to_add = tile_nums[tile_idx]

            modifiable_map = (jnp.isin(init_map, non_num_tiles) & (init_map != tile_enum.BORDER)).ravel()
            probs = modifiable_map / jnp.sum(modifiable_map)
            add_idxs = jax.random.choice(rng, n_map_cells, shape=(tiles_to_add,), p=probs, replace=False)
            
            # Adjust the map
            init_map = init_map.ravel().at[add_idxs].set(tile_idx).reshape(map_shape)

            return (rng, init_map), None

        tile_idxs = np.arange(len(tile_enum))
        # _, init_map = jax.lax.scan(adjust_tile_nums, (rng, init_map), tile_idxs)[0]
        for tile_idx in tile_idxs:
            (rng, init_map), _ = add_num_tiles((rng, init_map), tile_idx)

    return MapData(init_map, actual_map_shape)


class Problem:
    tile_size = np.int8(16)
    stat_weights: chex.Array
    metrics_enum: IntEnum
    ctrl_metrics: chex.Array
    stat_trgs: chex.Array
    ctrl_threshes: chex.Array = None
    queued_ctrl_trgs: chex.Array = None

    def __init__(self, map_shape, ctrl_metrics, pinpoints):
        self.map_shape = map_shape
        self.metric_bounds = self.get_metric_bounds(map_shape)
        self.ctrl_metrics = np.array(ctrl_metrics, dtype=int)
        self.ctrl_metrics_mask = np.array([i in ctrl_metrics for i in range(len(self.stat_trgs))])


        if self.ctrl_threshes is None:
            self.ctrl_threshes = np.zeros(len(self.stat_trgs))

        self.max_loss = get_max_loss(self.stat_weights, self.stat_trgs, 
                                     self.ctrl_threshes, self.metric_bounds)

        # Dummy control observation to placate jax tree map during minibatch creation (FIXME?)
        self.ctrl_metric_obs_idxs = np.array([0]) if len(self.ctrl_metrics) == 0 else self.ctrl_metrics

        self.metric_names = [metric.name for metric in self.metrics_enum]

        self.queued_ctrl_trgs = jnp.zeros(len(self.metric_names))  # dummy value to placate jax
        self.has_queued_ctrl_trgs = False

        # Make sure we don't generate pinpoint tiles if they are being treated as such
        tile_probs = np.array(self.tile_probs, dtype=np.float32)
        if self.tile_nums is not None and pinpoints:
            for tile in self.tile_enum:
                if self.tile_nums[tile] > 0:
                    tile_probs[tile] = 0
                # Normalize to make tile_probs sum to 1
            tile_probs = tile_probs / np.sum(tile_probs)
        self.tile_probs = tuple(tile_probs)

    @partial(jax.jit, static_argnames=("self", "randomize_map_shape", "empty_start", "pinpoints"))
    def gen_init_map(self, rng, randomize_map_shape=False, empty_start=False, pinpoints=False):
        return gen_init_map(rng, self.tile_enum, self.map_shape, self.tile_probs,
                            randomize_map_shape=randomize_map_shape, empty_start=empty_start, tile_nums=self.tile_nums,
                            pinpoints=pinpoints)

    def get_metric_bounds(self, map_shape):
        raise NotImplementedError

    def get_stats(self, env_map: chex.Array, prob_state: ProblemState):
        raise NotImplementedError

    # def queue_ctrl_trgs(self, queued_state, ctrl_trgs):
    #     queued_state = queued_state.replace(queued_ctrl_trgs=ctrl_trgs, has_queued_ctrl_trgs=True)
    #     return queued_state

    def init_graphics(self):
        self.graphics = jnp.array([np.array(g) for g in self.graphics])
        # Load TTF font (Replace 'path/to/font.ttf' with the actual path)
        self.render_font = font = ImageFont.truetype("./fonts/AcPlus_IBM_VGA_9x16-2x.ttf", 20)

        ascii_chars_to_ints = {}
        self.ascii_chars_to_ims = {}

        self.render_font_shape = (16, 9)

        # Loop over a range of ASCII characters (here, printable ASCII characters from 32 to 126)
        # for i in range(0, 127):
        #     char = chr(i)

        #     # Create a blank RGBA image
        #     image = Image.new("RGBA", self.render_font_shape, (0, 0, 0, 0))

        #     # Get drawing context
        #     draw = ImageDraw.Draw(image)

        #     # Draw text
        #     draw.text((0, 0), char, font=font, fill=(255, 255, 255, 255))

        #     ascii_chars_to_ints[char] = i
        #     char_im = np.array(image)
        #     self.ascii_chars_to_ims[char] = char_im

    def observe_ctrls(self, prob_state: ProblemState):
        if self.n_agents > 1:
            obs = jnp.zeros((len(self.metrics_enum), self.n_agents))
            obs = jnp.where(self.ctrl_metrics_mask[..., None], jnp.sign(prob_state.ctrl_trgs - prob_state.stats[..., None]), obs)
        else:
            obs = jnp.zeros(len(self.metrics_enum))
            obs = jnp.where(self.ctrl_metrics_mask, jnp.sign(prob_state.ctrl_trgs - prob_state.stats), obs)
        # Return a vector of only the metrics we're controlling
        obs = obs[self.ctrl_metric_obs_idxs]
        return obs

    def gen_rand_ctrl_trgs(self, rng, actual_map_shape):
        metric_bounds = self.get_metric_bounds(actual_map_shape)
        # Randomly sample some control targets
        if self.n_agents > 1:
            vmap_gen = jax.vmap(gen_ctrl_trgs, in_axes=(None, 0))
            new_keys = jax.random.split(rng, self.n_agents)
            ctrl_trgs = jnp.where(
                jnp.expand_dims(self.ctrl_metrics_mask, 1),
                vmap_gen(metric_bounds, new_keys).T, #Transpose because VMAP outputs [n_agents, num_ctrl_metrics]
                self.stat_trgs # while this is defined as [num_ctrl_metrics, n_agents]
                )
        else:
            ctrl_trgs =  jnp.where(
                self.ctrl_metrics_mask,
                gen_ctrl_trgs(metric_bounds, rng),
                self.stat_trgs,
            )
        return ctrl_trgs

    def reset(self, env_map: chex.Array, rng, queued_state, actual_map_shape):
        if self.n_agents > 1:
            queued_state = queued_state.replace(ctrl_trgs=jnp.tile(queued_state.ctrl_trgs, (3,1)).T)
        
        ctrl_trgs = jax.lax.select(
            queued_state.has_queued_ctrl_trgs,
            queued_state.ctrl_trgs,
            self.gen_rand_ctrl_trgs(rng, actual_map_shape),
        )
        
        state = self.get_curr_stats(env_map)
        state = state.replace(
            ctrl_trgs=ctrl_trgs,
        )

        reward = None
        return reward, state

    def step(self, env_map: chex.Array, state: ProblemState):
        new_state = self.get_curr_stats(env_map=env_map)
        reward = get_reward(new_state.stats, state.stats, self.stat_weights, state.ctrl_trgs, self.ctrl_threshes)
        new_state = new_state.replace(
            ctrl_trgs=state.ctrl_trgs,
        )
        return reward, new_state

    def get_curr_stats(self, env_map: chex.Array) -> ProblemState:
        raise NotImplementedError

    def draw_path(self, lvl_img, env_map, border_size, path_coords_tpl, tile_size):
        # path_coords_tpl is a tuple of (1) array of of path coordinates
        assert len(path_coords_tpl) == 1
        lvl_img = draw_path(prob=self, lvl_img=lvl_img, env_map=env_map, border_size=border_size, 
                  path_coords=path_coords_tpl[0], tile_size=tile_size)
        return lvl_img


def draw_path(prob, lvl_img, env_map, border_size, path_coords, tile_size,
              im_idx=-1):
    # Path, if applicable
    tile_img = prob.graphics[im_idx]

    def draw_path_tile(carry):
        path_coords, lvl_img, i = carry
        y, x = path_coords[i]
        tile_type = env_map[y + border_size[0]][x + border_size[1]]
        empty_tile = int(Tiles.EMPTY)

        # og_tile = lvl_img[(y + border_size[0]) * tile_size:(y + border_size[0] + 1) * tile_size,
        #                     (x + border_size[1]) * tile_size:(x + border_size[1] + 1) * tile_size, :]
        og_tile = jax.lax.dynamic_slice(lvl_img, ((y + border_size[0]) * tile_size, (x + border_size[1]) * tile_size, 0),
                                        (tile_size, tile_size, 4))
        new_tile_img = jnp.where(tile_img[..., -1:] == 0, og_tile, tile_img)

        # Only draw path tiles on top of empty tiles
        lvl_img = jax.lax.cond(
            tile_type == empty_tile,
            lambda: jax.lax.dynamic_update_slice(lvl_img, new_tile_img,
                                            ((y + border_size[0]) * tile_size, (x + border_size[1]) * tile_size, 0)),
            lambda: lvl_img,)
                            
        return (path_coords, lvl_img, i+1)

    def cond(carry):
        path_coords, _, i = carry
        return jnp.all(path_coords[i] != jnp.array((-1, -1)))
        # return jnp.all(path_coords[i:i+env.prob.max_path_len+1] != jnp.array(-1, -1))
        # result = jnp.any(
        #     jax.lax.dynamic_slice(path_coords, (i, 0), (prob.max_path_len+1, 2)) != jnp.array((-1, -1))
        # )
        # return result

    i = 0
    _, lvl_img, _ = jax.lax.while_loop(
        cond, draw_path_tile, (path_coords, lvl_img, i))

    return lvl_img
