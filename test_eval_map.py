import os

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from envs.pcgrl_env import PCGRLEnv, PCGRLEnvParams, PCGRLEnvState, RepEnum
from envs.pcgrl_env import ProbEnum
from envs.probs.dungeon import DungeonTiles


# map_name = 'binary_board_20240122_151126'
map_name = 'binary_board_20240122_165728'
with open(os.path.join('user_defined_freezies', f"{map_name}.npy"), 'rb') as f:
    board = jnp.load(f) + 1

env_params = PCGRLEnvParams(
    problem=ProbEnum.DUNGEON,
    representation=RepEnum.NARROW,
)
board = board.at[-1, -10].set(DungeonTiles.DOOR.value)
board = board.at[0, 0].set(DungeonTiles.KEY.value)
board = board.at[1, 0].set(DungeonTiles.BAT.value)
board = board.at[3, 0].set(DungeonTiles.BAT.value)
board = board.at[4, 0].set(DungeonTiles.BAT.value)
env = PCGRLEnv(env_params)
env.prob.init_graphics()
rng = jax.random.PRNGKey(0)
obs, env_state = env.reset(rng, env_params)
env_state = env_state.replace(env_map=board)
prob_state = env.prob.get_curr_stats(env_map=env_state.env_map)
env_state = env_state.replace(prob_state=prob_state)
im = env.render(env_state)
# Save image to disk
im = Image.fromarray(np.array(im).astype(np.uint8))
im.save(os.path.join('user_defined_freezies', f'{map_name}.png'))
print(board.shape)
print(board)