""" Wrappers for use with jaxmarl baselines. """
import math
import os
import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial

# from gymnax.environments import environment, spaces
from gymnax.environments.spaces import Box as BoxGymnax, Discrete as DiscreteGymnax
from typing import Dict, Optional, List, Tuple, Union
from envs.pcgrl_env import PCGRLEnvState, PCGRLEnvParams
from envs.probs.problem import get_loss
from marl.environments.spaces import Box, Discrete, MultiDiscrete
from marl.environments.multi_agent_env import EnvParams, MultiAgentEnv, State
from marl.model import ScannedRNN

from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict
from jax_utils import stack_leaves

def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)

def load_params(filename:Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


# @struct.dataclass
# class MultiAgentLogState(LogEnvState):
#     pass


class JaxMARLWrapper(object):
    """Base class for all jaxmarl wrappers."""

    def __init__(self, env: MultiAgentEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    # def _batchify(self, x: dict):
    #     x = jnp.stack([x[a] for a in self._env.agents])
    #     return x.reshape((self._env.n_agents, -1))

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class MALogWrapper(JaxMARLWrapper):
    """Log the episode returns and lengths.
    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, PCGRLEnvState]:
        obs, env_state = self._env.reset(key)
        env_state: PCGRLEnvState
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.n_agents,)),
            jnp.zeros((self._env.n_agents,)),
            jnp.zeros((self._env.n_agents,)),
            jnp.zeros((self._env.n_agents,)),
            # env_state.step_idx,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
        params: Optional[PCGRLEnvParams] = None,
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key=key, state=state.env_state, action=action, params=params,
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
            # timestep=state.timestep + 1,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.n_agents,), ep_done)
        return obs, state, reward, done, info


@struct.dataclass
class MALossLogEnvState:
    log_env_state: LogEnvState
    loss: float

class MALossLogWrapper(MALogWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: MultiAgentEnv):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey) -> Tuple[chex.Array, MultiAgentEnv]:
        obs, log_env_state = super().reset(key)
        state = MALossLogEnvState(log_env_state, jnp.inf)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        action: Union[int, float],
        params: Optional[PCGRLEnvParams] = None,
    ) -> Tuple[chex.Array, State, float, bool, dict]:
        obs, log_env_state, reward, done, info = super().step(
            key, state.log_env_state, action, params=params)
        loss = get_loss(log_env_state.env_state.prob_state.stats, 
                        self._env.prob.stat_weights, 
                        self._env.prob.stat_trgs,
                        self._env.prob.ctrl_threshes, 
                        self._env.prob.metric_bounds)
        # Normalize the loss to be in [0, 1]
        loss = loss / self._env.prob.max_loss

        state = MALossLogEnvState(
            log_env_state = log_env_state,
            loss = loss,
        )
        return obs, state, reward, done, info


    
class MultiAgentWrapper(JaxMARLWrapper):
    def __init__(self, env, env_params: PCGRLEnvParams):
        super().__init__(env)
        self.agents = [f'agent_{i}' for i in range(env.n_agents)]
        self._env.agents = self.agents
        self.flatten_obs = env_params.flatten_obs

        self.observation_spaces = {i: env.observation_space(env_params) for i in self.agents}
        self.action_spaces = {i: env.action_space(env_params) for i in self.agents}

        # TODO: Don't flatten like this!
        self.observation_spaces = {i: Box(low=-np.inf, high=np.inf, shape=(math.prod(space.shape),))
                                   for i, space in self.observation_spaces.items()}
        
        self.world_state_size = math.prod(self.observation_spaces[self.agents[0]].shape)

    def gen_dummy_obs(self, config):
        ac_init_x = (
            jnp.zeros((1, config.n_envs, self.observation_space(self.agents[0]).shape[0])),
            jnp.zeros((1, config.n_envs)),
            jnp.zeros((1, config.n_envs, self.action_space(self.agents[0]).n)),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config.n_envs, config.hidden_dims[0])
        return ac_init_x, ac_init_hstate

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    def process_observation(self, obs):
        # TODO: Don't flatten like this!
        new_obs = {}
        for i, agent in enumerate(self.agents):
            # TODO: Observe flat/scalar tings!
            # new_obs[agent] = jnp.concat((obs.map_obs[i].flatten(), obs.flat_obs.flatten()), axis=0)
            if self.flatten_obs:
                new_obs[agent] = obs.map_obs[i].flatten()
            else:
                new_obs[agent] = jax.tree.map(lambda x: x[i], obs)
                new_obs[agent] = new_obs[agent].replace(
                    flat_obs=jnp.expand_dims(new_obs[agent].flat_obs, axis=0)
                )
        if self.flatten_obs:
            new_obs['world_state'] = jnp.stack([new_obs[agent] for agent in self.agents])
        else:
            new_obs['world_state'] = stack_leaves(new_obs.values())
        return new_obs

    def reset(self, key):
        obs, state = self._env.reset(key)
        obs = self.process_observation(obs)

        return obs, state

    def step(self, key, state, action, params):
        obs, state, reward, done, info = self._env.step_ma(key, state, action, params)
        obs = self.process_observation(obs)
        return obs, state, reward, done, info

    def get_avail_actions(self, state: PCGRLEnvState):
        return {
            agent: jnp.ones((self.action_space(agent).n,)) for agent in self.agents
        }


class JaxMARLWrapper(object):
    """Base class for all jaxmarl wrappers."""

    def __init__(self, env: MultiAgentEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    # def _batchify(self, x: dict):
    #     x = jnp.stack([x[a] for a in self._env.agents])
    #     return x.reshape((self._env.num_agents, -1))

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])


class LogWrapper(JaxMARLWrapper):
    """Log the episode returns and lengths.
    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.n_agents,)),
            jnp.zeros((self._env.n_agents,)),
            jnp.zeros((self._env.n_agents,)),
            jnp.zeros((self._env.n_agents,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key=key, state=state.env_state, action=action
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info

class MPELogWrapper(LogWrapper):
    """ Times reward signal by number of agents within the environment,
    to match the on-policy codebase. """
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        rewardlog = jax.tree_map(lambda x: x*self._env.num_agents, reward)  # As per on-policy codebase
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(rewardlog)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info

@struct.dataclass
class SMAXLogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    won_episode: int
    returned_episode_returns: float
    returned_episode_lengths: int
    returned_won_episode: int


class SMAXLogWrapper(JaxMARLWrapper):
    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        state = SMAXLogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: SMAXLogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        batch_reward = self._batchify_floats(reward)
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1
        new_won_episode = (batch_reward >= 1.0).astype(jnp.float32)
        state = SMAXLogEnvState(
            env_state=env_state,
            won_episode=new_won_episode * (1 - ep_done),
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
            returned_won_episode=state.returned_won_episode * (1 - ep_done)
            + new_won_episode * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_won_episode"] = state.returned_won_episode
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info


def get_space_dim(space):
    # get the proper action/obs space from Discrete-MultiDiscrete-Box spaces
    if isinstance(space, (DiscreteGymnax, Discrete)):
        return space.n
    elif isinstance(space, (BoxGymnax, Box, MultiDiscrete)):
        return np.prod(space.shape)
    else:
        print(space)
        raise NotImplementedError('Current wrapper works only with Discrete/MultiDiscrete/Box action and obs spaces')


class CTRolloutManager(JaxMARLWrapper):
    """
    Rollout Manager for Centralized Training of with Parameters Sharing. Used by JaxMARL Q-Learning Baselines.
    - Batchify multiple environments (the number of parallel envs is defined by batch_size in __init__).
    - Adds a global state (obs["__all__"]) and a global reward (rewards["__all__"]) in the env.step returns.
    - Pads the observations of the agents in order to have all the same length.
    - Adds an agent id (one hot encoded) to the observation vectors.

    By default:
    - global_state is the concatenation of all agents' observations.
    - global_reward is the sum of all agents' rewards.
    """

    def __init__(self, env: MultiAgentEnv, batch_size:int, training_agents:List=None, preprocess_obs:bool=True):
        
        super().__init__(env)
        
        self.batch_size = batch_size

        # the agents to train could differ from the total trainable agents in the env (f.i. if using pretrained agents)
        # it's important to know it in order to compute properly the default global rewards and state
        self.training_agents = self.agents if training_agents is None else training_agents  
        self.preprocess_obs = preprocess_obs  

        # TOREMOVE: this is because overcooked doesn't follow other envs conventions
        if len(env.observation_spaces) == 0:
            self.observation_spaces = {agent:self.observation_space() for agent in self.agents}
        if len(env.action_spaces) == 0:
            self.action_spaces = {agent:env.action_space() for agent in self.agents}
        
        # batched action sampling
        self.batch_samplers = {agent: jax.jit(jax.vmap(self.action_space(agent).sample, in_axes=0)) for agent in self.agents}

        # assumes the observations are flattened vectors
        self.max_obs_length = max(list(map(lambda x: get_space_dim(x), self.observation_spaces.values())))
        self.max_action_space = max(list(map(lambda x: get_space_dim(x), self.action_spaces.values())))
        self.obs_size = self.max_obs_length + len(self.agents)

        # agents ids
        self.agents_one_hot = {a:oh for a, oh in zip(self.agents, jnp.eye(len(self.agents)))}
        # valid actions
        self.valid_actions = {a:jnp.arange(u.n) for a, u in self.action_spaces.items()}
        self.valid_actions_oh ={a:jnp.concatenate((jnp.ones(u.n), jnp.zeros(self.max_action_space - u.n))) for a, u in self.action_spaces.items()}

        # custom global state and rewards for specific envs
        if 'smax' in env.name.lower():
            self.global_state = lambda obs, state: obs['world_state']
            self.global_reward = lambda rewards: rewards[self.training_agents[0]]*10
        elif 'overcooked' in env.name.lower():
            self.global_state = lambda obs, state:  jnp.concatenate([obs[agent].ravel() for agent in self.agents], axis=-1)
            self.global_reward = lambda rewards: rewards[self.training_agents[0]]

    
    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, key):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_reset, in_axes=0)(keys)

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, key, states, actions):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_step, in_axes=(0, 0, 0))(keys, states, actions)

    @partial(jax.jit, static_argnums=0)
    def wrapped_reset(self, key):
        obs_, state = self._env.reset(key)
        if self.preprocess_obs:
            obs = jax.tree_util.tree_map(self._preprocess_obs, {agent:obs_[agent] for agent in self.agents}, self.agents_one_hot)
        else:
            obs = obs_
        obs["__all__"] = self.global_state(obs_, state)
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def wrapped_step(self, key, state, actions):
        if 'hanabi' in self._env.name.lower():
            actions = jax.tree_util.tree_map(lambda x:jnp.expand_dims(x, 0), actions)
        obs_, state, reward, done, infos = self._env.step(key, state, actions)
        if self.preprocess_obs:
            obs = jax.tree_util.tree_map(self._preprocess_obs, {agent:obs_[agent] for agent in self.agents}, self.agents_one_hot)
            obs = jax.tree_util.tree_map(lambda d, o: jnp.where(d, 0., o), {agent:done[agent] for agent in self.agents}, obs) # ensure that the obs are 0s for done agents
        else:
            obs = obs_
        obs["__all__"] = self.global_state(obs_, state)
        reward["__all__"] = self.global_reward(reward)
        return obs, state, reward, done, infos

    @partial(jax.jit, static_argnums=0)
    def global_state(self, obs, state):
        return jnp.concatenate([obs[agent] for agent in self.agents], axis=-1)
    
    @partial(jax.jit, static_argnums=0)
    def global_reward(self, reward):
        return jnp.stack([reward[agent] for agent in self.training_agents]).sum(axis=0) 
    
    def batch_sample(self, key, agent):
        return self.batch_samplers[agent](jax.random.split(key, self.batch_size)).astype(int)

    @partial(jax.jit, static_argnums=0)
    def _preprocess_obs(self, arr, extra_features):
        # flatten
        arr = arr.flatten()
        # pad the observation vectors to the maximum length
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, max(0, self.max_obs_length - arr.shape[-1]))]
        arr = jnp.pad(arr, pad_width, mode='constant', constant_values=0)
        # concatenate the extra features
        arr = jnp.concatenate((arr, extra_features), axis=-1)
        return arr
