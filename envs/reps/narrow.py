from functools import partial
import math
from typing import Tuple

import chex
from flax import struct
from gymnax.environments import spaces
import jax
import jax.numpy as jnp
import numpy as np

from envs.utils import Tiles
from envs.reps.representation import (Representation, RepresentationState,
                                      get_ego_obs)


@struct.dataclass
class NarrowRepresentationState(RepresentationState):
    pos: chex.Array
    agent_coords: chex.Array
    n_valid_agent_coords: int

    
@partial(jax.jit, static_argnames=('tile_enum', 'act_shape', 'rand_coords', 'max_board_scans'))
def gen_agent_coords(
    frz_map: chex.Array,
    tile_enum: Tiles,
    act_shape: Tuple[int, int],
    rand_coords: bool,
    max_board_scans: int,
    rng: chex.PRNGKey = None,
):
    # TODO: If using larger action patches, ignore patches that are all frozen.
    
    if act_shape != (1, 1):
        # This is static. Not going to be recomputed/recompiled each function
        # call... right? D:
        # TODO: Factor this out, only needs to be called once.
        agent_coords = np.argwhere(np.ones(frz_map.shape, dtype=np.uint8))
                                    # size=math.prod(frz_map.shape))
        # Filter out coordinates so that agent with larger action shapes do 
        # minimal redundant builds (note that they may still make some 
        # overlapping builds near the edges).
        m, n = act_shape
        agent_coords = agent_coords[
            (agent_coords[:, 0] % m == 0) &
            (agent_coords[:, 1] % n == 0)]
        n_valid_agent_coords = np.int32(len(agent_coords))
    else:
        # Skip frozen tiles.
        agent_coords = jnp.argwhere(frz_map == 0, size=math.prod(frz_map.shape), fill_value=-1)
        n_valid_agent_coords = jnp.sum(frz_map == 0)
    if rand_coords:
        agent_coords = jnp.concat([jax.random.permutation(
            rng, agent_coords) for rng in jax.random.split(rng, math.ceil(max_board_scans))], axis=0)
        # Now sort the agent coords (N, 2) so that all (-1, -1) are at the end
        coords_are_valid = jnp.where(agent_coords[:, 0] >= 0, True, False)
        shuffled_coords_indices = jnp.argsort(coords_are_valid, descending=True)
        agent_coords = agent_coords[shuffled_coords_indices]
        n_valid_agent_coords = n_valid_agent_coords * math.ceil(max_board_scans)
    return agent_coords, n_valid_agent_coords


class NarrowRepresentation(Representation):
    def __init__(self, env_map: chex.Array, rf_shape: Tuple[int, int],
                 act_shape: Tuple[int, int], tile_enum: Tiles, max_board_scans: int, pinpoints: bool, tile_nums: Tuple[int],
                 rand_coords: bool):
        super().__init__(tile_enum=tile_enum, rf_shape=rf_shape,
                         act_shape=act_shape, pinpoints=pinpoints, tile_nums=tile_nums)
        self.rf_shape = np.array(rf_shape)
        self.rf_off = int(max(np.ceil(self.rf_shape - 1) / 2))
        self.max_board_scans = max_board_scans
        self.max_steps = np.uint32(np.prod(env_map.shape) * max_board_scans)
        self.num_tiles = len(tile_enum)
        self.builds = jnp.array(self.editable_tile_enum)
        self.rand_coords = rand_coords

        self.act_shape = act_shape

    def step(self, env_map: chex.Array, action: int,
             rep_state: NarrowRepresentationState, step_idx: int, agent_id: int = 0):
        action = action[..., 0]
        b = self.builds[action]
        pos_idx = step_idx % rep_state.n_valid_agent_coords
        new_pos = rep_state.agent_coords[pos_idx]
        new_env_map = jax.lax.dynamic_update_slice(env_map, b, new_pos)

        # map_changed = jnp.logical_not(jnp.array_equal(new_env_map, env_map))
        rep_state = NarrowRepresentationState(
            pos=new_pos,
            agent_coords=rep_state.agent_coords,
            n_valid_agent_coords=rep_state.n_valid_agent_coords,
        )

        return new_env_map, rep_state

    def reset(self, frz_map: chex.Array = None, rng: chex.PRNGKey = None):
        agent_coords, n_valid_agent_coords = gen_agent_coords(
            frz_map, self.tile_enum, self.act_shape, self.rand_coords, self.max_board_scans, rng)
        pos = agent_coords[0]

        return NarrowRepresentationState(
            pos=pos,
            agent_coords=agent_coords,
            n_valid_agent_coords=n_valid_agent_coords)
    
    @property
    def action_space(self) -> spaces.Discrete:
        # return spaces.Discrete(len(self.tile_enum) - 1)
        return spaces.Discrete(self.n_editable_tiles)

    get_obs = get_ego_obs


if __name__ == "__main__":
    def test_narrow_agent_coords_never_on_frozen():
        """Initialize with a partially frozen map and ensure agent coords exclude frozen tiles.

        This test uses act_shape=(1, 1) and rand_coords=False to exercise the
        frz_map filtering path in gen_agent_coords.
        """
        key = jax.random.PRNGKey(0)

        # Small map: 5x5, with a plus-shaped frozen region in the center
        H, W = 5, 5
        env_map = jnp.full((H, W), Tiles.EMPTY, dtype=jnp.int32)
        frz_map = jnp.zeros((H, W), dtype=bool)
        frz_map = frz_map.at[2, :].set(True)   # middle row frozen
        frz_map = frz_map.at[:, 2].set(True)   # middle column frozen

        rep = NarrowRepresentation(
            env_map=env_map,
            rf_shape=(3, 3),
            act_shape=(1, 1),
            tile_enum=Tiles,
            max_board_scans=1,
            pinpoints=False,
            tile_nums=tuple(0 for _ in range(len(Tiles))),
            rand_coords=False,
        )

        rep_state = rep.reset(frz_map=frz_map, rng=key)

        # Number of valid coordinates should match count of non-frozen tiles
        expected_valid = int((~frz_map).sum())
        n_valid = int(jnp.array(rep_state.n_valid_agent_coords))
        assert n_valid == expected_valid, (
            f"Expected {expected_valid} valid coords, got {n_valid}")

        # All coordinates up to n_valid must be valid and not -1
        coords = np.array(rep_state.agent_coords)
        for i in range(n_valid):
            y, x = map(int, coords[i])
            assert y >= 0 and x >= 0, f"Invalid coord at index {i}: {(y, x)}"
            assert not bool(frz_map[y, x]), f"Frozen coord selected at index {i}: {(y, x)}"

        # Also check the initial position isn't frozen
        y0, x0 = map(int, np.array(rep_state.pos))
        assert not bool(frz_map[y0, x0]), f"Initial pos on frozen tile: {(y0, x0)}"

        # Step several times and ensure position never lands on a frozen tile
        env_map_cur = env_map
        for t in range(min(20, n_valid)):
            env_map_cur, rep_state = rep.step(env_map_cur, action=0, rep_state=rep_state, step_idx=t, rng=key)
            yt, xt = map(int, np.array(rep_state.pos))
            assert not bool(frz_map[yt, xt]), f"Step {t} pos on frozen tile: {(yt, xt)}"

        print("test_narrow_agent_coords_never_on_frozen: PASS")

    test_narrow_agent_coords_never_on_frozen()
