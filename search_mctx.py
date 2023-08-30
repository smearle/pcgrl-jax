from timeit import default_timer as timer
import functools

import chex
import jax
from jax import numpy as jnp
from config import TrainConfig
import mctx
from utils import gymnax_pcgrl_make, init_config

from envs.pcgrl_env import PCGRLEnvState


Reward = chex.Array

config = TrainConfig(
    representation='narrow',
    max_board_scans=1.0,
    act_shape=(1, 1),
)
config = init_config(config)
env, env_params = gymnax_pcgrl_make(config.env_name, config)


def policy_function(env_state: PCGRLEnvState, rng: chex.PRNGKey) -> chex.Array:
    # logits = env.sample_action(rng)
    logits = jnp.ones(env.action_shape(), dtype=jnp.float32).reshape(-1)
    # logits = jnp.ones_like(action, dtype=jnp.float32).reshape(-1)
    return logits

def value_function(env_state: PCGRLEnvState, rng: chex.PRNGKey) -> chex.Array:
    return env_state.reward
    # return rollout(env_state, rng).astype(jnp.float32)

def rollout(env_state: PCGRLEnvState, rng: chex.PRNGKey) -> Reward:

    def cond(a):
        env_state, key = a
        return env_state.done

    def step(a):
        env_state, key = a
        key, _key = jax.random.split(key)
        # action = env.sample_action(_key).reshape(-1)
        action = env.sample_action(key)
        key, _key = jax.random.split(key)
        env_state, reward, done = env_step(_key, action, env_state)
        return env_state, key
    leaf, key = jax.lax.while_loop(cond, step, (env_state, rng))
    return leaf.reward

def root_fn(env_state: PCGRLEnvState, rng_key: chex.PRNGKey) -> mctx.RootFnOutput:
    return mctx.RootFnOutput(
        prior_logits=policy_function(env_state, rng_key),
        value=value_function(env_state, rng_key),
        # We will use the `embedding` field to store the environment.
        embedding=env_state,
    )

def env_step(key, action: chex.Array, env_state: PCGRLEnvState):
    obs, env_state, reward, done, info = env.step(
        key, env_state, action, env_params
    )
    return env_state, reward, done

def recurrent_fn(params, rng_key, action, embedding):
    # Extract the environment from the embedding.
    env_state = embedding

    # breakpoint()
    # action = jnp.unravel_index(action, (1,) + env.action_shape()[:-1])
    # NOTE: Restricted to single-agent representation with action patch of size 1
    action = action[None, None, None, None, ...]

    # Play the action
    env_state, reward, done = env_step(rng_key, action, env_state)

    # Create the new MCTS node.
    recurrent_fn_output = mctx.RecurrentFnOutput(
        # reward for playing `action`
        reward=reward,
        # discount explained in the next section
        # discount=jnp.where(done, 0, -1).astype(jnp.float32),
        # default discount
        discount=1,
        # prior for the new state
        prior_logits=policy_function(env_state, rng_key),
        # value for the new state
        value=jnp.where(done, 0, value_function(env_state, rng_key)).astype(jnp.float32),
    )

    # Return the new node and the new environment.
    return recurrent_fn_output, env_state

@functools.partial(jax.jit, static_argnums=(2,))
def run_mcts(rng_key: chex.PRNGKey, env_state: PCGRLEnvState, num_simulations: int) -> chex.Array:
    batch_size = 400
    key1, key2 = jax.random.split(rng_key)
    policy_output = mctx.muzero_policy(
        # params can be used to pass additional data to the recurrent_fn like neural network weights
        params=None,

        rng_key=key1,

        # create a batch of environments (in this case, a batch of size 1)
        root=jax.vmap(root_fn, (None, 0))(env_state, jax.random.split(key2, batch_size)),

        # automatically vectorize the recurrent_fn
        recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),

        num_simulations=num_simulations,

        # we limit the depth of the search tree to 42, since we know that Connect Four can't last longer
        max_depth=env.rep.max_steps,

        # our value is in the range [-1, 1], so we can use the min_max qtransform to map it to [0, 1]
        qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),

        # Dirichlet noise is used for exploration which we don't need in this example (we aren't training)
        dirichlet_fraction=0.0,
    )
    return policy_output


key = jax.random.PRNGKey(0)
key, reset_key = jax.random.split(key)
obs, env_state = env.reset(reset_key, env_params)
start_time = timer()
policy_output = run_mcts(key, env_state, 4000)
print(f"Time taken: {timer() - start_time}")
search_tree = policy_output.search_tree
breakpoint()

