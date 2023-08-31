
from enum import IntEnum

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Stats(IntEnum):
    pass


@struct.dataclass
class ProblemState:
    stats: chex.Array
    ctrl_trgs: chex.Array


def get_reward(stats, old_stats, stat_weights, stat_trgs):
    reward = jnp.abs(stat_trgs - old_stats) - jnp.abs(stat_trgs - stats)
    reward = jnp.where(stat_trgs == jnp.inf, stats - old_stats, reward)
    reward = jnp.where(stat_trgs == -jnp.inf, old_stats - stats, reward)
    reward *= stat_weights
    reward = jnp.sum(reward)
    return reward


def get_reward_old(stats, old_stats, stat_weights, stat_trgs):
    reward = 0
    for stat, val in stats.items():
        old_val = old_stats[stat]
        trg = stat_trgs[stat]
        if trg == 'max':
            reward += (val - old_val) * stat_weights[stat]
        elif trg == 'min':
            reward += (old_val - val) * stat_weights[stat]
        elif isinstance(trg, tuple):
            old_loss = min(abs(old_val - trg[0]), abs(old_val - trg[1]))
            new_loss = min(abs(val - trg[0]), abs(val - trg[1]))
            reward += (old_loss - new_loss) * stat_weights[stat]
        else:
            reward += (abs(old_val - trg)
                       - abs(val - trg)) * stat_weights[stat]
    return reward


class Problem:
    tile_size = np.int8(16)

    def __init__(self):
        # Dummy control observation to placate jax tree map during minibatch creation (FIXME?)
        self.ctrl_metric_obs_idxs = np.array([0]) if len(self.ctrl_metrics) == 0 else self.ctrl_metrics

    def get_stats(self, env_map: chex.Array, prob_state: ProblemState):
        raise NotImplementedError

    def init_graphics(self):
        # Load TTF font (Replace 'path/to/font.ttf' with the actual path)
        font = ImageFont.truetype("./fonts/AcPlus_IBM_VGA_9x16-2x.ttf", 20)

        ascii_chars_to_ints = {}
        self.ascii_chars_to_ims = {}

        # Loop over a range of ASCII characters (here, printable ASCII characters from 32 to 126)
        for i in range(0, 127):
            char = chr(i)

            # Create a blank RGBA image
            image = Image.new("RGBA", (20, 20), (0, 0, 0, 0))

            # Get drawing context
            draw = ImageDraw.Draw(image)

            # Draw text
            draw.text((0, 0), char, font=font, fill=(255, 255, 255, 255))

            # Resize image to 16x16
            # image = image.resize((16, 16), Image.ANTIALIAS)

            # Save image
            # image.save(f"char_images/{i}.png")

            ascii_chars_to_ints[char] = i
            self.ascii_chars_to_ims[char] = np.array(image)

    def observe_ctrls(self, prob_state: ProblemState):
        obs = jnp.zeros(len(self.metrics_enum))
        obs = jnp.where(self.ctrl_metrics_mask, jnp.sign(prob_state.ctrl_trgs - prob_state.stats), obs)
        # Return a vector of only the metrics we're controlling
        obs = obs[self.ctrl_metric_obs_idxs]
        return obs