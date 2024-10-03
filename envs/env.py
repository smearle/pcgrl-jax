
from functools import partial
from typing import Optional, Tuple, Union
import chex
from gymnax import EnvParams, EnvState

from gymnax.environments.environment import Environment as GymnaxEnvironment
import jax

class Environment(GymnaxEnvironment):
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        agent_id: int,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key, state, action, params, agent_id=agent_id
        )
        obs_re, state_re = self.reset_env(key_reset, params, queued_state=state.queued_state)
        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        # obs = jax.lax.select(done, obs_re, obs_st)
        # Generalizing this to flax dataclass observations
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st
        )
        return obs, state, reward, done, info


    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None,
        queued_state=None,
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        if queued_state is None:
            queued_state = self.dummy_queued_state
        obs, state = self.reset_env(key, params, queued_state)
        return obs, state
