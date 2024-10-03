import copy
import json
import os
import shutil
import sys
import time
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
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


from torch.utils.tensorboard import SummaryWriter


def create_message_box(text):
    # 텍스트를 줄 단위로 나눕니다.
    lines = text.split('\n')

    # 각 줄의 길이를 구하고, 가장 긴 줄의 길이를 찾습니다.
    max_length = max(len(line) for line in lines)

    # 메시지 박스의 길이를 가장 긴 줄의 길이에 맞춰서 설정합니다.
    box_length = max_length + 6  # ### 양쪽 3자리씩 차지함

    # 메시지 박스를 구성합니다.
    top_bottom_border = "#" * box_length
    middle_lines = [f"### {line.ljust(max_length)} ###" for line in lines]

    # 메시지 박스를 문자열로 반환
    return f"{top_bottom_border}\n" + "\n".join(middle_lines) + f"\n{top_bottom_border}"

def get_textfile_tail(log_path, tail: int = 60) -> str:
    with open(log_path, 'rb') as f:
        # Move to the end of the file
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        block_size = 1024
        data = []
        while file_size > 0 and len(data) < tail:
            if file_size - block_size > 0:
                f.seek(-block_size, os.SEEK_CUR)
            else:
                f.seek(0)
                block_size = file_size
            chunk = f.read(block_size).splitlines()
            data = chunk + data
            file_size -= block_size
            f.seek(file_size, os.SEEK_SET)

        # Trim the list to the last 'tail' lines
        if len(data) > tail:
            data = data[-tail:]
        logs = [line.decode('utf-8') for line in data]

    return '\n'.join(logs)


class Experiment:

    def __init__(self, kwargs):

        self.kwargs = kwargs

        self._config_yaml = kwargs.config_path

        self._env = str(path.abspath(kwargs.env))
        self._num_envs = kwargs.num_envs
        self._no_graphics = kwargs.no_graphics
        self._resume = kwargs.resume
        self._overwrite = kwargs.overwrite

        self._reward_generation_path = kwargs.reward_generator_path

        self._root_path = kwargs.workspace

        self._current_workspace = kwargs.workspace

        self._config_dict = self.read_yaml(self._config_yaml)
        self._reward_generation_settings = self._config_dict.get('reward_generation_settings', {})

        self._run_id = self._config_dict['checkpoint_settings']['run_id']

        self._stage = 'reward_generation'
        self._iteration = 1

        self._reward_function_log_filename = 'RewardFunctionLog.csv'
        self._current_reward_function_filename = None
        self._feedback_path = None
        self._experiment_path = path.abspath(path.join(self._root_path, self._run_id))
        self._experiment_log_path = path.join(self._experiment_path, 'experiment.log')
        self._create_experiment_dir()
        self._setup_logger()
        self._log_step = 0
        self._log_sleep_time = kwargs.log_sleep_time
        self.log_and_print(str(kwargs))
        self._start_watch_log()

        if self._resume:
            self.load_state()

    @property
    def reward_function_log_path(self):
        return path.join(self._experiment_path, self._reward_function_log_filename)

    def _create_experiment_dir(self):
        # make directory
        if not path.exists(self._root_path):
            os.makedirs(self._root_path)

        self.log_and_print(f"Creating experiment directory: {self._experiment_path}")

        try:
            os.makedirs(self._experiment_path, exist_ok=self._resume or self._overwrite)
        except FileExistsError:
            self.log_and_print(f"Experiment directory already exists: {self._experiment_path}")
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
            self.log_and_print(f"Prompt directory already exists: {dest_dir}")
            pass

    def _copy_example(self):
        """Copies the prompt log file to the experiment directory."""
        source_dir = path.join(path.dirname(__file__), 'example')
        dest_dir = path.join(self._experiment_path, 'example')

        try:
            shutil.copytree(source_dir, dest_dir)
        except FileExistsError:
            self.log_and_print(f"Prompt directory already exists: {dest_dir}")
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
            self.log_and_print(f"Reward functions directory already exists: {self.reward_functions_dir}")
            raise

    def _create_feedback_dir(self):
        try:
            os.makedirs(self.feedback_dir, exist_ok=self._resume or self._overwrite)
        except FileExistsError:
            self.log_and_print(f"Feedback directory already exists: {self.feedback_dir}")
            raise

    def read_yaml(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def generate_reward_function(self, bypass_reward_function: str = None):
        """Generates a reward function using the reward generator script."""

        tgt_path = os.path.join(self._experiment_path, 'GeneratedSkill.csv')
        api_key = self._get_env_args(self._config_dict, 'pcgGptApiKey')
        gpt_model = self._get_env_args(self._config_dict, 'pcgGptModel')
        feedback_type = self._get_env_args(self._config_dict, 'feedbackType')
        if feedback_type == "t-SNE" and gpt_model != "gpt-4o":
            self.log_and_print(create_message_box("You should use gpt-4o for vision feedback"))
            self._exit()
            exit(1)

        message = ''
        if self._iteration == 1:
            self._reference_csv = 'random_dataset.txt'
        else:
            self._reference_csv = tgt_path

        if self._iteration != 1:
            if feedback_type == "t-SNE":
                self._feedback_path = os.path.join(self.feedback_dir, f"outer_{self._iteration}", feedback_type, "t-SNE.png")
            elif feedback_type == "statistics":
                self._feedback_path = os.path.join(self.feedback_dir, f"outer_{self._iteration}", feedback_type, "statistics.json")

            feedback_args_dict = {
                'shared_storage_path': self._experiment_path,
                'postfix': f"outer_{self._iteration}",
                'feedback_type': feedback_type,
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
            'api_key': api_key,
            'shared_storage_path': self._experiment_path,
            'postfix': f"reward_outer_{self._iteration}",
            'reward_functions_dir': 'RewardFunctions',
            'gpt_model': gpt_model,
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

        self.log_and_print(create_message_box(f"Generating reward function (iteration {self._iteration})"))

        args_list = [item for key, value in args_dict.items() if value is not None for item in
                     (f'--{key}', str(value))]


        is_bypassed = (bypass_reward_function is not None) and \
                        (self._iteration == 1)

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
                    self.log_and_print(output, end='')

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

    def remove_reward_generation_settings(self, config):
        """Removes 'reward_generation_settings' from the configuration dictionary."""
        if 'reward_generation_settings' in config:
            del config['reward_generation_settings']
        return config

    def replace_workspace_paths(self, config, old_path, new_path):
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and old_path in value:
                    config[key] = value.replace(old_path, new_path)
                elif isinstance(value, dict) or isinstance(value, list):
                    config[key] = self.replace_workspace_paths(value, old_path, new_path)
        elif isinstance(config, list):
            for index, item in enumerate(config):
                if isinstance(item, str) and old_path in item:
                    config[index] = item.replace(old_path, new_path)
                elif isinstance(item, dict) or isinstance(item, list):
                    config[index] = self.replace_workspace_paths(item, old_path, new_path)
        return config

    def _get_env_args_index(self, config_dict: dict, arg_name: str) -> int:
        """Returns the index of the argument in the list of environment arguments."""

        _arg_name = f'--{arg_name}'

        try:
            return config_dict['env_settings']['env_args'].index(_arg_name)
        except ValueError:
            return -1

    def _get_env_args(self, config_dict: dict, env_args: str) -> str:
        idx = self._get_env_args_index(config_dict, env_args)

        if idx == -1:
            return None
        else:
            return config_dict['env_settings']['env_args'][idx + 1]

    def _set_env_args(self, config_dict: dict, env_args: str, value: str):
        """Sets the environment arguments for the mlagents-learn command."""
        idx = self._get_env_args_index(config_dict, env_args)
        if idx != -1:
            config_dict['env_settings']['env_args'][idx + 1] = value
        else:
            config_dict['env_settings']['env_args'].extend([f'--{env_args}', value])

    @property
    def _iteration_run_id(self):
        return f"iteration_{self._iteration}"

    def run_mlagents(self):
        """Runs the mlagents-learn command with the given parameters."""

        config = copy.deepcopy(self._config_dict)

        # Save the modified YAML configuration to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_config_file:
            if 'reward_generation_settings' in config:
                del config['reward_generation_settings']

            self._set_env_args(config, 'runId', f"{self._run_id}_{self._iteration_run_id}")
            self._set_env_args(config, 'logPath', path.join(self._experiment_path, self._iteration_run_id))

            if self._current_reward_function_filename is not None:
                self._set_env_args(config, 'pcgGptFirstRewardFunction',
                                   path.join(self.reward_functions_dir, self._current_reward_function_filename[:-3], self._current_reward_function_filename))

            config['checkpoint_settings']['run_id'] = self._iteration_run_id
            config['checkpoint_settings']['results_dir'] = self._experiment_path

            config = self.replace_workspace_paths(config, '/workspace/results', path.abspath(self._current_workspace))
            yaml.dump(config, temp_config_file)

            temp_config_path = temp_config_file.name

        additional = list()
        if path.exists(path.join(self._experiment_path, self._iteration_run_id)):
            additional.append('--resume')

        if os.name != 'nt':  # not Windows

            # Start mlagents-learn process
            mlagents_process = subprocess.Popen(
                ['mlagents-learn', temp_config_path, '--env', self._env, '--num-envs', str(self._num_envs),
                 '--no-graphics', *additional],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding=None)

        else:
            site_packages_path = site.getsitepackages()[0]

            mlagents_process = subprocess.Popen(
                ['python', '-m', 'mlagents.trainers.learn',
                 temp_config_path, '--env', self._env, '--num-envs', str(self._num_envs), '--no-graphics',
                 *additional],
                cwd=site_packages_path,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')

        directory_watcher_path = os.path.join(path.dirname(__file__), 'directory_watcher.py')

        self.log_and_print(self._experiment_path, self._iteration_run_id)

        # Start directory_watcher.py process
        watcher_process = subprocess.Popen(
            ['python3', directory_watcher_path, '--directory',
             f'{path.join(self._experiment_path, self._iteration_run_id)}', '--yaml_path', temp_config_path,
             '--start_sleep_time', '20'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def read_output(process, name):
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.rstrip()
                    self.log_and_print(f"[{name}] {output}")

        # Create threads to read the outputs
        mlagents_thread = threading.Thread(target=read_output, args=(mlagents_process, 'mlagents'))
        watcher_thread = threading.Thread(target=read_output, args=(watcher_process, 'watcher'))

        mlagents_thread.start()
        watcher_thread.start()

        mlagents_thread.join()

        self.log_and_print(create_message_box(f"ML-Agents process for iteration {self._iteration} has finished."))

        # Clean up temporary file
        os.remove(temp_config_path)

        del mlagents_thread
        del watcher_thread

        mlagents_process.kill()
        watcher_process.kill()

    def collect_results(self, iteration_run_id):

        src_path = os.path.join(self._experiment_path, iteration_run_id, 'SharedStorage', 'GeneratedSkill.csv')
        tgt_path = os.path.join(self._experiment_path, 'GeneratedSkill.csv')

        try:
            src_df = pd.read_csv(src_path)
            # add iteration column in the first
            src_df['LLM.RewardIteration'] = self._iteration

        except FileNotFoundError:
            self.log_and_print(create_message_box(f"Results not found for iteration {self._iteration}"))
            return

        try:
            tgt_df = pd.read_csv(tgt_path)
            # Concatenate the source and target dataframes
            combined_df = pd.concat([tgt_df, src_df], ignore_index=True)
        except FileNotFoundError:
            # If the target file doesn't exist, use the source dataframe
            combined_df = src_df

        # Save the combined dataframe to the target path
        combined_df.to_csv(tgt_path, index=False)

        self.log_and_print(create_message_box(f"Results collected for iteration {self._iteration}"))

    # 파일 분석
    def analyze_output(self, output):

        self.log_and_print("Output analyzed")

    def _check_experiment_dir_exists(self):
        if path.exists(self._experiment_path):
            self.log_and_print(
                create_message_box(f"Experiment directory already exist: {self._experiment_path}. Exiting."))
            exit(1)

    def save_state(self):
        # target variables: iteration, current_reward_function_filename
        serialize_items = ['_stage', '_iteration', '_current_reward_function_filename', '_log_step']

        with open(path.join(self._experiment_path, 'state.yaml'), 'w') as file:
            yaml.dump({item: getattr(self, item) for item in serialize_items}, file)

        self.log_and_print(create_message_box("State saved."))

    def load_state(self):
        self.log_and_print(create_message_box("Loading state from the previous experiment."))

        try:
            with open(path.join(self._experiment_path, 'state.yaml'), 'r') as file:
                state = yaml.safe_load(file)

                for key, value in state.items():
                    setattr(self, key, value)
        except FileNotFoundError:
            self.log_and_print(create_message_box("State file not found. Exiting."))

    def run(self):

        for i_inner in range(self._iteration, self._reward_generation_settings.get('outer_loop', 1) + 1):

            while True:

                if self._stage == 'reward_generation':
                    self.log_and_print(
                        create_message_box(f"Generating reward function for iteration {self._iteration}"))
                    reward_function_path, message = self.generate_reward_function(self._reward_generation_settings.get('first_iter_reward', None))
                    if reward_function_path is None:
                        self.send_message_to_slack(f'Reward function generation failed.\n{message}')

                        self.log_and_print(create_message_box(f"Reward function generation failed.\n{message}\nExiting."))
                        self._exit()
                        exit(1)
                    else:
                        self._current_reward_function_filename = reward_function_path

                    self._stage = 'mlagents'
                elif self._stage == 'mlagents':
                    self.log_and_print(create_message_box(f"Running ML-Agents for iteration {self._iteration}"))
                    # Run ML-Agents
                    self.run_mlagents()

                    self._stage = 'results_collection'
                elif self._stage == 'results_collection':
                    self.log_and_print(create_message_box(f"Collecting results for iteration {self._iteration}"))
                    # Collect results
                    self.collect_results(self._iteration_run_id)

                    self._stage = 'analysis'
                elif self._stage == 'analysis':
                    self.log_and_print(create_message_box(f"Analyzing results for iteration {self._iteration}"))
                    # Analyze results
                    self.analyze_output(self._iteration_run_id)

                    self._stage = 'iteration_finished'
                elif self._stage == 'iteration_finished':
                    self._iteration += 1
                    self._stage = 'reward_generation'
                    break

            self.save_state()

        self.log_and_print(create_message_box("Experiment finished."))
        self._exit()
        exit()
    def _exit(self):
        self._exit_event.set()
        self.watch_thread.join()


    def _setup_logger(self):
        """Sets up the logger for the experiment."""
        self.logger = logging.getLogger('ExperimentLogger')
        self.logger.setLevel(logging.DEBUG)

        # Create file handler which logs even debug messages
        log_file_path = path.join(self._experiment_path, 'experiment.log')
        fh = logging.FileHandler(log_file_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_and_print(self, message, end='\n'):
        print(message, end=end)
        if hasattr(self, 'logger'):
            if message.startswith('###'):
                message = '\n' + message
            self.logger.info(message.strip())

    def _start_watch_log(self):
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)

        self.writer = SummaryWriter(self.tensorboard_log_dir)
        self.log_and_print(f"Tensorboard writer created at {self.tensorboard_log_dir}")
        self.log_and_print("Directory watcher initialized.")

        self._exit_event = threading.Event()
        self.watch_thread = threading.Thread(target=self.watch_experiment_log)
        self.watch_thread.start()


    def watch_experiment_log(self):
        while not self._exit_event.is_set():
            try:
                content = get_textfile_tail(self._experiment_log_path)

                content_str = (f'***Path: {self._experiment_log_path}***\n'
                               f'```json\n{content}\n```')

                self.writer.add_text('ExperimentLog', content_str, self._log_step)
                self.writer.flush()
                self.log_and_print(f"Writing experiment log to tensorboard (len: {len(content)})")

                self._log_step += 1
            except Exception as e:
                self.log_and_print(f"Failed writing experiment log to tensorboard:\n{e}")

            self._exit_event.wait(self._log_sleep_time)

    def send_message_to_slack(self, message_text:str):
        slack_channel = self._get_env_args(self._config_dict, 'slackChannel')
        slack_token = self._get_env_args(self._config_dict, 'slackToken')

        if slack_token is None or slack_channel is None:
            self.log_and_print("Slack token or channel not found. Skipping sending message to Slack.")
            return

        api_url = f"https://slack.com/api/chat.postMessage"

        new_message_text = message_text.replace("\"", "\\\"")

        # Constructing the message
        sb = list()
        sb.append("```")
        sb.append(f"Experiment: {self._run_id}")
        sb.append(f"Log Path: {self._experiment_path}")
        sb.append(f"Message: {new_message_text}")
        sb.append("```")

        message = "\n".join(sb).strip()

        print(f"Slack Message: \n{message}")

        slack_message = {
            "channel": slack_channel,
            "text": message
        }

        json_data = json.dumps(slack_message)

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {slack_token}"
        }

        response = requests.post(api_url, headers=headers, data=json_data)

        if response.status_code != 200:
            self.log_and_print(f"Failed to send message to Slack: {response.text}")
        else:
            # Parse JSON and check if "ok" is true
            response_json = response.json()
            if response_json.get("ok"):
                self.log_and_print(f"Message sent to Slack: {message_text}")
            else:
                self.log_and_print(f"Error in slack response: {response_json}")


def is_slack_enabled():
    # Implement your logic to check if Slack is enabled
    return True


def get_default_environment_path():
    system = platform.system()

    if system == 'Linux':
        return '/game/MMORPG.x86_64'
    elif system == 'Windows':
        return os.path.join('..', '..', '..', 'RaidEnv', 'Build', 'Win', 'MMORPG.exe')
    elif system == 'Darwin':  # Darwin은 macOS의 내부 시스템 이름입니다.
        return os.path.join('..', '..', '..', 'RaidEnv', 'Build', 'MacOS', 'MMORPG.app')

    else:
        raise OSError('Unsupported operating system')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML-Agents with a specified configuration file.")
    parser.add_argument('config_path', type=str, default='debug.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--env', type=str, default=get_default_environment_path(),
                        help='Path to the executable environment file')
    parser.add_argument('--num_envs', type=int, default=2, help='Number of environments to run in parallel.')
    parser.add_argument('--no_graphics', action='store_true', help='Run the environment without graphics.')
    parser.add_argument('--workspace', type=str, default='./workspace', help='Path to the workspace directory.')
    parser.add_argument('--log_sleep_time', type=int, default=600,
                        help='Time to sleep between checking reward functions (in seconds)')
    parser.add_argument('--reward_generator_path', type=str, default='generate_reward.py',
                        help='Path to the reward generator directory.')
    parser.add_argument('--overwrite ', action='store_true', dest='overwrite',
                        help='Overwrite the existing experiment directory.')
    parser.add_argument('--resume', action='store_true', dest='resume',
                        help='Resume the experiment from the existing directory.')

    args = parser.parse_args()

    experiment = Experiment(args)
    experiment.run()