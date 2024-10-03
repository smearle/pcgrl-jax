from os.path import basename

import hydra
import json
import os
import shutil
import copy
from datetime import datetime
from os import path
import subprocess
import argparse

import requests
import yaml
import tempfile
import threading
import platform
import pandas as pd
import logging
import os
import site
import platform
import pprint

from conf.config import TrainConfig
from eval import main_eval
from pcgrllm.paths import init_config


from pcgrllm.stage import Stage


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="orbax")
warnings.filterwarnings("ignore", category=FutureWarning, module="jax")

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))


logging.getLogger('hydra').setLevel(logging.WARNING)
logging.getLogger('absl').setLevel(logging.WARNING)


class Experiment:

    def __init__(self, config: TrainConfig):
        self.config = config

        if config.overwrite and os.path.exists(self.config.exp_dir):
            shutil.rmtree(self.config.exp_dir)

        os.makedirs(self.config.exp_dir, exist_ok=True)

        self.initialize()
        self._setup_logger()

        self.logging(pprint.pformat(self.config, indent=4), level=logging.INFO)

    def initialize(self):
        self._iteration = 1
        self._stage = Stage.StartIteration
        self._current_reward_function_filename = None

    @property
    def _experiment_path(self):
        return self.config.exp_dir

    def log_with_prefix(self, message, level=logging.DEBUG):
        """Logs a message with a formatted prefix."""
        info_dict = {
            'outer_loop': getattr(self, '_iteration', -1),
            'inner_loop': getattr(self, 'current_inner', -1),
            'n_inner': getattr(self, 'n_inner', -1),
            'trial': getattr(self, '_current_trial', -1),
            'trial_count': getattr(self, 'trial_count', -1),
        }

        # Define the prefix format
        prefix = '[ol: {outer_loop}, il: {inner_loop} / {n_inner}, trial: {trial} / {trial_count}]'.format(**info_dict)

        # Split the message by line breaks and log each line with the prefix
        message = str(message)
        for line in message.splitlines():
            formatted_message = f'{prefix} {line}'
            self.logger.log(level, formatted_message)



    @property
    def reward_function_log_path(self):
        return path.join(self._experiment_path, self._reward_function_log_filename)

    def _create_experiment_dir(self):
        # make directory
        if not path.exists(self._root_path):
            os.makedirs(self._root_path)

        self.logging(f"Creating experiment directory: {self._experiment_path}")

        try:
            os.makedirs(self._experiment_path, exist_ok=self._resume or self._overwrite)
        except FileExistsError:
            self.logging(f"Experiment directory already exists: {self._experiment_path}")
            raise

        self._create_reward_functions_dir()
        self._create_feedback_dir()
        self._copy_prompt()
        self._copy_example()

    def _copy_prompt(self):
        """Copies the prompt log file to the experiment directory."""
        source_dir = path.join(path.dirname(__file__), 'prompt')
        dest_dir = path.join(self._experiment_path, 'prompt')

        try:
            shutil.copytree(source_dir, dest_dir)
        except FileExistsError:
            self.logging(f"Prompt directory already exists: {dest_dir}")
            pass

    def _copy_example(self):
        """Copies the prompt log file to the experiment directory."""
        source_dir = path.join(path.dirname(__file__), 'example')
        dest_dir = path.join(self._experiment_path, 'example')

        try:
            shutil.copytree(source_dir, dest_dir)
        except FileExistsError:
            self.logging(f"Prompt directory already exists: {dest_dir}")
            pass

    @property
    def reward_functions_dir(self):
        return path.join(self._experiment_path, 'RewardFunctions')

    @property
    def feedback_dir(self):
        return path.join(self._experiment_path, 'Feedback')

    @property
    def tensorboard_log_dir(self):
        return path.join(self._experiment_path, 'experiment_logs')

    def _create_reward_functions_dir(self):
        try:
            os.makedirs(self.reward_functions_dir, exist_ok=self._resume or self._overwrite)
        except FileExistsError:
            self.logging(f"Reward functions directory already exists: {self.reward_functions_dir}")
            raise

    def _create_feedback_dir(self):
        try:
            os.makedirs(self.feedback_dir, exist_ok=self._resume or self._overwrite)
        except FileExistsError:
            self.logging(f"Feedback directory already exists: {self.feedback_dir}")
            raise


    def generate_reward_function(self, bypass_reward_function: str = None):
        """Generates a reward function using the reward generator script."""

        tgt_path = os.path.join(self._experiment_path, 'GeneratedSkill.csv')

        # feedback_type = self._get_env_args(self._config_dict, 'feedbackType')
        # if feedback_type == "t-SNE" and gpt_model != "gpt-4o":
        #     self.logging(create_message_box("You should use gpt-4o for vision feedback"))
        #     self._exit()
        #     exit(1)

        message = ''
        if self._iteration == 1:
            self._reference_csv = 'random_dataset.txt'
        else:
            self._reference_csv = tgt_path

        if self._iteration != 1:
            # if feedback_type == "t-SNE":
            #     self._feedback_path = os.path.join(self.feedback_dir, f"outer_{self._iteration}", feedback_type, "t-SNE.png")
            # elif feedback_type == "statistics":
            #     self._feedback_path = os.path.join(self.feedback_dir, f"outer_{self._iteration}", feedback_type, "statistics.json")

            feedback_args_dict = {
                'shared_storage_path': self._experiment_path,
                'postfix': f"outer_{self._iteration}",
                # 'feedback_type': feedback_type,
                'skill_log_csv': self._reference_csv
            }
            feedback_args_list = [item for key, value in feedback_args_dict.items() if value is not None for item in
                         (f'--{key}', str(value))]

            generate_feedback_py = os.path.join(path.dirname(__file__), 'generate_feedback.py')
            command_line = ['python', generate_feedback_py, *feedback_args_list]

            process = subprocess.Popen(command_line,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            while True:
                error = process.stderr.readline()
                if error == '' and process.poll() is not None:
                    break
                if error:
                    message = f'Fail:{error.strip()}'
                    return None, message

        arbitrary_dataset = path.abspath(path.join(path.dirname(__file__), 'example', 'arbitrary_dataset.txt'))

        if self._current_reward_function_filename is not None:
            previous_reward_function = self._current_reward_function_filename
        else:
            previous_reward_function = None

        args_dict = {
            # 'api_key': api_key,
            'shared_storage_path': self._experiment_path,
            'postfix': f"reward_outer_{self._iteration}",
            'reward_functions_dir': 'RewardFunctions',
            # 'gpt_model': gpt_model,
            'gpt_max_token': 4096,
            'verbose': None,
            'previous_reward_function': previous_reward_function,
            'trial_count': self._reward_generation_settings['trial_count'],
            'n_inner': self._reward_generation_settings['inner_loop'],
            'n_outer': self._iteration,
            'reference_csv': self._reference_csv,
            'iteration_num': self._iteration,
            'feedback_path': self._feedback_path,
            'arbitrary_dataset': arbitrary_dataset,
        }

        self.logging(f"Generating reward function (iteration {self._iteration})")

        args_list = [item for key, value in args_dict.items() if value is not None for item in
                     (f'--{key}', str(value))]

        is_bypassed = (bypass_reward_function is not None) and (self._iteration == 1)



        if not is_bypassed:

            # Start of the 'generate_reward.py'

            generate_reward_py = os.path.join(path.dirname(__file__), 'generate_reward.py')
            command_line = ['python', generate_reward_py, *args_list]
            process = subprocess.Popen(command_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            is_success = False
            error_output = ''
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()

                if output == '' and error == '' and process.poll() is not None:
                    break

                if output:
                    if 'Done' in output:
                        is_success = True
                    self.logging(output, end='')

                if error:
                    error_output += error

            # End of the 'generate_reward.py'
        else:
            # Copy the bypass reward function to the reward functions directory
            src_path = bypass_reward_function

            reward_name = f"{args_dict['postfix']}_inner_{args_dict['n_inner']}"
            dir_path = os.path.join(self.reward_functions_dir, reward_name)

            # create directory if not exists
            os.makedirs(dir_path, exist_ok=True)

            tgt_path = os.path.join(dir_path, f"{reward_name}.py")
            shutil.copy(src_path, tgt_path)

            is_success = True
            error_output = ''

        message = 'Success' if is_success else f'Fail: {error_output.strip()}'

        reward_name = f"{args_dict['postfix']}_inner_{args_dict['n_inner']}.py"
        self.append_reward_generation_log(trial_num=0, result=message,
                                          previous_reward_function=previous_reward_function,
                                          current_reward_function=reward_name)

        if is_success:
            # copy the reward function as iteration_2.py
            if not is_bypassed:
                message = f"Reward function generated successfully: {reward_name}"
            else:
                message = f"Reward function bypassed: {reward_name}"
            # src_path = os.path.join(self.reward_functions_dir, f"reward_{args_dict['postfix']}.py")
            # tgt_path = os.path.join(self.reward_functions_dir, reward_name)
            # shutil.copy(src_path, tgt_path)

            return reward_name, message
        return None, message


    def append_reward_generation_log(self, result: str, trial_num: int, previous_reward_function: str,
                                     current_reward_function: str) -> None:
        # append dataframe
        row = {
            'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            'Iteration': self._iteration,
            'InstanceUUID': platform.node(),
            'Trial': trial_num,
            'Result': result,
            'PreviousFileName': previous_reward_function,
            'CurrentFileName': current_reward_function,
            'Academy.TotalStepCount': 0,
            'Academy.EpisodeCount': 0
        }

        try:
            df = pd.read_csv(self.reward_function_log_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        except FileNotFoundError:
            df = pd.DataFrame([row])

        df.to_csv(self.reward_function_log_path, index=False)

    def train_pcgrl(self):
        """Runs the mlagents-learn command with the given parameters."""

        config = copy.deepcopy(self.config)
        config.exp_dir = path.join(config.exp_dir, f'iteration_{self._iteration}')
        config.total_timesteps = 100 # TODO Debug
        os.makedirs(config.exp_dir, exist_ok=True)

        config.overwrite = False

        from train import main_noinit as train

        train(config)

        logging.debug(111)

        return True

        # Save the modified YAML configuration to a temporary file
        # with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_config_file:
        #     if 'reward_generation_settings' in config:
        #         del config['reward_generation_settings']
        #
        #     self._set_env_args(config, 'runId', f"{self._run_id}_{self._iteration_run_id}")
        #     self._set_env_args(config, 'logPath', path.join(self._experiment_path, self._iteration_run_id))
        #
        #     if self._current_reward_function_filename is not None:
        #         self._set_env_args(config, 'pcgGptFirstRewardFunction',
        #                            path.join(self.reward_functions_dir, self._current_reward_function_filename[:-3], self._current_reward_function_filename))
        #
        #     config['checkpoint_settings']['run_id'] = self._iteration_run_id
        #     config['checkpoint_settings']['results_dir'] = self._experiment_path
        #
        #     config = self.replace_workspace_paths(config, '/workspace/results', path.abspath(self._current_workspace))
        #     yaml.dump(config, temp_config_file)
        #
        #     temp_config_path = temp_config_file.name
        #
        # additional = list()
        # if path.exists(path.join(self._experiment_path, self._iteration_run_id)):
        #     additional.append('--resume')
        #
        # if os.name != 'nt':  # not Windows
        #
        #     # Start mlagents-learn process
        #     mlagents_process = subprocess.Popen(
        #         ['mlagents-learn', temp_config_path, '--env', self._env, '--num-envs', str(self._num_envs),
        #          '--no-graphics', *additional],
        #         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding=None)
        #
        # else:
        #     site_packages_path = site.getsitepackages()[0]
        #
        #     mlagents_process = subprocess.Popen(
        #         ['python', '-m', 'mlagents.trainers.learn',
        #          temp_config_path, '--env', self._env, '--num-envs', str(self._num_envs), '--no-graphics',
        #          *additional],
        #         cwd=site_packages_path,
        #         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        #
        # directory_watcher_path = os.path.join(path.dirname(__file__), 'directory_watcher.py')
        #
        # self.logging(self._experiment_path, self._iteration_run_id)
        #
        # # Start directory_watcher.py process
        # watcher_process = subprocess.Popen(
        #     ['python3', directory_watcher_path, '--directory',
        #      f'{path.join(self._experiment_path, self._iteration_run_id)}', '--yaml_path', temp_config_path,
        #      '--start_sleep_time', '20'],
        #     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        #
        # def read_output(process, name):
        #     while True:
        #         output = process.stdout.readline()
        #         if output == '' and process.poll() is not None:
        #             break
        #         if output:
        #             output = output.rstrip()
        #             self.logging(f"[{name}] {output}")
        #
        # # Create threads to read the outputs
        # mlagents_thread = threading.Thread(target=read_output, args=(mlagents_process, 'mlagents'))
        # watcher_thread = threading.Thread(target=read_output, args=(watcher_process, 'watcher'))
        #
        # mlagents_thread.start()
        # watcher_thread.start()
        #
        # mlagents_thread.join()
        #
        # self.logging(f"ML-Agents process for iteration {self._iteration} has finished.")
        #
        # # Clean up temporary file
        # os.remove(temp_config_path)
        #
        # del mlagents_thread
        # del watcher_thread
        #
        # mlagents_process.kill()
        # watcher_process.kill()

    def rollout_pcgrl(self, iteration_run_id):
        from eval import main_eval as run_eval

        config = copy.deepcopy(self.config)
        config.exp_dir = path.join(config.exp_dir, 'iteration_' + str(self._iteration))
        config.random_agent = False
        # config.INIT_CONFIG = False
        # get parametrer for eval

        run_eval(config)


    # 파일 분석
    def analyze_output(self, output):
        self.logging("Output analyzed")

    def save_state(self):
        # target variables: iteration, current_reward_function_filename
        serialize_items = ['_stage', '_iteration', '_current_reward_function_filename']

        with open(path.join(self._experiment_path, 'state.yaml'), 'w') as file:
            yaml.dump({item: getattr(self, item) for item in serialize_items}, file)


    def load_state(self):
        self.logging("Loading state from the previous experiment.")

        try:
            with open(path.join(self._experiment_path, 'state.yaml'), 'r') as file:
                state = yaml.safe_load(file)

                for key, value in state.items():
                    setattr(self, key, value)
        except FileNotFoundError:
            self.logging("State file not found. Exiting.")

    def run(self):

        self.logging("Running experiment", level=logging.DEBUG)

        while not self._stage is Stage.Done:

            self.logging(f"Current stage: {self._stage}", level=logging.DEBUG)

            if self._stage == Stage.StartIteration:
                self._stage = Stage.RewardGeneration

            elif self._stage == Stage.RewardGeneration:
                self.logging(
                    f"Generating reward function for iteration {self._iteration}", level=logging.INFO)
                # reward_function_path, message = self.generate_reward_function() # TODO (self._reward_generation_settings.get('first_iter_reward', None))
                #
                # if reward_function_path is None:
                #     self.logging(f"Reward function generation failed.\n{message}\nExiting.")
                #     self.exit("Reward function generation failed.")
                # else:
                #     self._current_reward_function_filename = reward_function_path


                self._stage = Stage.TrainPCGRL
            elif self._stage == Stage.TrainPCGRL:
                # Run ML-Agents
                self.train_pcgrl()

                self._stage = Stage.RolloutPCGRL
            elif self._stage == Stage.RolloutPCGRL:
                # Collect results
                self.rollout_pcgrl(self._iteration)

                self._stage = Stage.Analysis
            elif self._stage == Stage.Analysis:
                # Analyze results
                # self.analyze_output(self._iteration) # TODO
                self._stage = Stage.FinishIteration

            elif self._stage == Stage.FinishIteration:

                if self._iteration >= self.config.total_iterations:
                    self._stage = Stage.Done
                else:
                    self._iteration += 1
                    self._stage = Stage.StartIteration

            self.save_state()

        self.logging("Experiment finished.")

    def exit(self, message: str, code: int = 1):
        self.logging(message, level=logging.ERROR)
        exit(code)

    def _setup_logger(self):
        """Sets up the logger for the experiment."""
        self.logger = logging.getLogger(basename(__file__))
        self.logger.setLevel(logging.DEBUG)

        #
        # # Create file handler which logs even debug messages
        # log_file_path = path.join(self._experiment_path, 'experiment.log')
        # fh = logging.FileHandler(log_file_path, encoding='utf-8')
        # fh.setLevel(logging.DEBUG)
        #
        # # Create console handler with a higher log level
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.ERROR)
        #
        # # Create formatter and add it to the handlers
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # fh.setFormatter(formatter)
        # ch.setFormatter(formatter)
        #
        # # Add the handlers to the logger
        # self.logger.addHandler(fh)
        # self.logger.addHandler(ch)



    def logging(self, message, level=logging.DEBUG):
        info_dict = {
            'outer_loop': self.iteration_num if hasattr(self, 'iteration_num') else -1,
            'trial': self._current_trial if hasattr(self, '_current_trial') else -1,  # Assuming self._current_trial is a class attribute
            'trial_count': self.trial_count if hasattr(self, 'trial_count') else -1,
        }

        # Define the prefix format
        prefix = '[ol: {outer_loop}, trial: {trial} / {trial_count}]'.format(**info_dict)

        # Split the message by line breaks and log each line with the prefix
        message = str(message)
        for line in message.splitlines():
            formatted_message = f'{prefix} {line}'
            self.logger.log(level, formatted_message)





@hydra.main(version_base=None, config_path='./conf', config_name='train_pcgrllm')
def main(config: TrainConfig):
    init_config(config)



    experiment = Experiment(config)
    experiment.run()


if __name__ == "__main__":
    main()

