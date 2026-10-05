0# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train an agent to solve the classic CartPole swing up task.

Example command to run this script:
# Train in a harder setup.
python train_cartpole.py --gpu-id=0
# Train in an easy setup.
python train_cartpole.py --gpu-id=0 --easy
# Train a permutation invariant agent in a harder setup.
python train_cartpole.py --gpu-id=0 --pi --max-iter=20000 --pop-size=256 \
--center-lr=0.037 \
--std-lr=0.078 \
--init-std=0.082
# Train a permutation invariant agent in a harder setup with CMA-ES.
python train_cartpole.py --gpu-id=0 --pi --max-iter=20000 --pop-size=256 --cma
"""

import argparse
from dataclasses import dataclass
import imageio
import math
import os
import shutil
from typing import Tuple
import chex
import flax
import jax
import jax.numpy as jnp

from evojax import Trainer
from evojax.task.cartpole import CartPoleSwingUp
from evojax.policy import MLPPolicy
from evojax.policy import PermutationInvariantPolicy
from evojax.algo import PGPE
from evojax.algo import CMA
from evojax import util
from evojax.util import get_tensorboard_log_fn

from envs.pcgrl_env import PCGRLEnv, PCGRLEnvParams, PCGRLEnvState, flatten_obs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size', type=int, default=64, help='NE population size.')
    parser.add_argument(
        '--hidden-size', type=int, default=64, help='Policy hidden size.')
    parser.add_argument(
        '--num-tests', type=int, default=100, help='Number of test rollouts.')
    parser.add_argument(
        '--n-repeats', type=int, default=16, help='Training repetitions.')
    parser.add_argument(
        '--max-iter', type=int, default=1000, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=100, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=20, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument(
        '--center-lr', type=float, default=0.05, help='Center learning rate.')
    parser.add_argument(
        '--std-lr', type=float, default=0.1, help='Std learning rate.')
    parser.add_argument(
        '--init-std', type=float, default=0.1, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--easy', action='store_true', help='Easy mode.')
    parser.add_argument(
        '--pi', action='store_true', help='Permutation invariant policy.')
    parser.add_argument(
        '--cma', action='store_true', help='Training with CMA-ES.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


@flax.struct.dataclass
class PCGRLEvoEnvState:
    obs: chex.Array
    state: PCGRLEnvState


class PCGRLEvoWrapper(PCGRLEnv):
    def __init__(self, test=False, harder=False, env_params: PCGRLEnvParams = None):
        super().__init__(env_params=env_params)

        # FIXME: PCGRLEnv problem seems to add flat observation of shape (1,) even when nothing to observe
        self.obs_shape = (math.prod(self.observation_space(env_params).shape) + 1,)

        self.act_shape = (math.prod(self.rep.action_space().shape),)
        self.multi_agent_training = False

        def reset_fn(key) -> PCGRLEvoEnvState:
            obs, state = PCGRLEnv.reset(self, key)
            obs = flatten_obs(obs)
            next_key, key = jax.random.split(key)
            return PCGRLEvoEnvState(obs=obs, state=state)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state: PCGRLEvoEnvState, action):

            # Only get away with this because PCGRLEnv does not use the key each step. Otherwise kinda sketchy of evojax
            # to make this assumption.
            key = jax.random.PRNGKey(0)

            obs, state, reward, done, info = PCGRLEnv.step(self, key, state.state, action[None,None])
            obs = flatten_obs(obs)
            return PCGRLEvoEnvState(obs=obs, state=state), reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> PCGRLEvoEnvState:
        return self._reset_fn(key)

    def step(self,
             state: PCGRLEvoEnvState,
             action: jnp.ndarray) -> Tuple[PCGRLEvoEnvState, jnp.ndarray, jnp.ndarray]:
        action = action.astype(int)
        return self._step_fn(state, action)


def main_evolve(config):
    hard = not config.easy

    env_params = PCGRLEnvParams()
    train_task = PCGRLEvoWrapper(test=False, harder=hard, env_params=env_params)

    # # Reset the environment
    # key = jax.random.PRNGKey(seed=0)
    # # state = train_task._reset_fn(key)
    # state = train_task.reset(key)
    # # Take 100 steps in the environment
    # for _ in range(100):
    #     # Sample random actions
    #     key, _ = jax.random.split(key)
    #     action = jax.random.uniform(key, shape=train_task.act_shape)
    #     # Take a step
    #     state, reward, done = train_task.step(state, action)
    #     # state, reward, done = train_task._step_fn(state, action)
    #     # Render the environment
    #     im = train_task.render(state, 0)

    log_dir = './log/pcgrl_{}'.format('hard' if hard else 'easy')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='PCGRL', log_dir=log_dir, debug=config.debug)

    logger.info('EvoJAX PCGRL')
    logger.info('=' * 30)

    env_params = PCGRLEnvParams()
    train_task = PCGRLEvoWrapper(test=False, harder=hard, env_params=env_params)
    test_task = PCGRLEvoWrapper(test=True, harder=hard, env_params=env_params)
    if config.pi:
        policy = PermutationInvariantPolicy(
            act_dim=test_task.act_shape[0],
            hidden_dim=config.hidden_size,
        )
    else:
        policy = MLPPolicy(
            input_dim=train_task.obs_shape[0],
            hidden_dims=[config.hidden_size] * 2,
            output_dim=train_task.act_shape[0],
        )
    if config.cma:
        solver = CMA(
            pop_size=config.pop_size,
            param_size=policy.num_params,
            init_stdev=config.init_std,
            seed=config.seed,
            logger=logger,
        )
    else:
        solver = PGPE(
            pop_size=config.pop_size,
            param_size=policy.num_params,
            optimizer='adam',
            center_learning_rate=config.center_lr,
            stdev_learning_rate=config.std_lr,
            init_stdev=config.init_std,
            logger=logger,
            seed=config.seed,
        )

    try:
        log_scores_fn = get_tensorboard_log_fn(log_dir=os.path.join(log_dir, "tb_logs"))
    except ImportError as e:
        logger.warning(e)

        def log_scores_fn(i, scores, stage):  # noqa
            pass
    
    model_dir = log_dir if os.path.exists(log_dir) else None

    # Train.
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=config.n_repeats,
        n_evaluations=config.num_tests,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
        log_scores_fn=log_scores_fn,
        model_dir=model_dir,
    )
    trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(log_dir, 'best.npz')
    tar_file = os.path.join(log_dir, 'model.npz')
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)

    # Generate a GIF to visualize the policy.
    best_params = trainer.solver.best_params[None, :]
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    act_fn = jax.jit(policy.get_actions)
    rollout_key = jax.random.PRNGKey(seed=0)[None, :]

    images = []
    task_s = task_reset_fn(rollout_key)
    policy_s = policy_reset_fn(task_s)
    train_task.init_graphics()
    r_state = jax.tree_map(lambda x: x[0], task_s.state)
    images.append(train_task.render(r_state))
    done = False
    step = 0
    while not done:
        act, policy_s = act_fn(task_s, best_params, policy_s)
        task_s, r, d = step_fn(task_s, act)
        step += 1
        done = bool(d[0])
        if step % 5 == 0:
            r_state = jax.tree_map(lambda x: x[0], task_s.state)
            images.append(train_task.render(r_state))

    gif_file = os.path.join(
        log_dir, 'pcgrl_{}.gif'.format('hard' if hard else 'easy'))

    imageio.v3.imwrite(
        gif_file,
        images,
        duration=20
    )
    # images[0].save(
    #     gif_file, save_all=True, append_images=images[1:], duration=40, loop=0)
    logger.info('GIF saved to {}'.format(gif_file))


if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main_evolve(configs)