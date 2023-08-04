
from enum import IntEnum

import chex
from flax import struct


class Stats(IntEnum):
    pass


@struct.dataclass
class ProblemState:
    pass


def get_reward(stats, old_stats, stat_weights, stat_trgs):
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
            reward += (abs(old_val - trg) - abs(val - trg)) * stat_weights[stat]
    return reward


class Problem:

    def get_stats(self, env_map: chex.Array, prob_state: ProblemState):
        raise NotImplementedError

    def init_graphics(self):
        raise NotImplementedError