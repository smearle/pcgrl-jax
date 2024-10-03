import copy
import datetime
import os, re, ast, sys, time, argparse, json, pickle
import shutil
import multiprocessing
import astor
import pandas as pd
import numpy as np
from os import path
import tempfile
from os.path import abspath, basename
import subprocess
import traceback
import logging

from example.utils.preprocessing import preprocessing

logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

def make_sure_package(package_name, package):
    try:
        globals()[package_name] = __import__(package_name)
    except ImportError:
        print(f'installing {package}')
        os.system(f'pip install {package}')
        globals()[package_name] = __import__(package_name)


make_sure_package('openai', 'openai==1.33.0')
make_sure_package('astor', 'astor')
make_sure_package('mlagents', 'mlagents')
make_sure_package('pkg_resources', 'pkg_resources')


def change_package_version(package_name, version=None):
    try:
        if version:
            installed_version = pkg_resources.get_distribution(package_name).version
            if installed_version != version:
                print(
                    f'{package_name} version {installed_version} is installed, but version {version} is required. Installing the required version.')
                raise ImportError
        globals()[package_name] = __import__(package_name)
    except ImportError:
        package = package_name if version is None else f'{package_name}=={version}'
        print(f'Installing {package}')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        globals()[package_name] = __import__(package_name)
        print(f'{package_name} has been installed and imported.')

change_package_version('importlib_metadata', '4.4.0')


from llm_client.llm import ChatContext, UnifiedLLMClient
from llm_client.utils import *

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))


class RewardGenerator:
    def __init__(self, config: dict):


        self.api_key = config.get('api_key')
        self.shared_storage_path = config.get('shared_storage_path', '.')
        self.postfix = config.get('postfix', time.strftime("%Y-%m-%d-%H-%M-%S"))
        self.reward_functions_dir = config.get('reward_functions_dir', 'RewardFunctions')
        self.gpt_model = config.get('gpt_model', 'gpt-3.5-turbo')
        self.gpt_max_token = config.get('gpt_max_token', 1024)
        self.verbose = config.get('verbose', False)
        self.n_inner = config.get('n_inner', 3)

        self.current_inner = config.get('current_inner', 1)
        self.logging(f"Current Inner: {self.current_inner}", logging.DEBUG)

        self.n_outer = config.get('n_outer', 3)
        self._current_trial = 0
        self.trial_count = config.get('trial_count', 3)
        self.iteration_num = config.get('iteration_num', 1)
        self.reference_csv = config.get('reference_csv', 'random_dataset.txt')
        self.arbitrary_dataset = config.get('arbitrary_dataset', 'arbitrary_dataset.txt')
        self.file_path = path.join(self.shared_storage_path, 'prompt')
        self.example_path = path.join(self.shared_storage_path, 'example')
        if self.reference_csv == 'random_dataset.txt':
            self.reference_csv = path.join(self.example_path, self.reference_csv)
        self.current_state_path = path.abspath(path.join(self.shared_storage_path, 'example', 'testState.json'))
        self.reward_function_path = path.join(self.shared_storage_path, self.reward_functions_dir,
                                         (str(self.postfix) + '_inner_' + str(self.n_inner)))
        self.initial_system = file_to_string(path.join(self.file_path, "initial_system.txt"))
        self.initial_user = file_to_string(path.join(self.file_path, "initial_user.txt"))
        self.task_description = file_to_string(path.join(self.file_path, "task_description.txt"))
        self.second_user = file_to_string(path.join(self.file_path, "second_user.txt"))

        self.reward_template = file_to_string(path.join(self.example_path, "compute_reward_template.py"))
        self.sampled_data_example = file_to_string(path.join(self.example_path, "sampled_data_example.txt"))
        self.reward_example = file_to_string(path.join(self.example_path, "compute_reward_example.py"))

        # previous 나중에 변경하기
        self.previous_reward_function = config.get('previous_reward_function', 'compute_reward_example.py')
        if self.previous_reward_function == 'compute_reward_example.py':
            self.previous_reward_function = file_to_string(path.join(self.example_path, self.previous_reward_function))
        else:
            self.previous_reward_function = file_to_string(path.join(self.shared_storage_path, self.reward_functions_dir, self.previous_reward_function[:-3], self.previous_reward_function))

        os.makedirs(self.reward_function_path, exist_ok=True)
        self._ensure_utility_files(self.reward_function_path)



        self.initial_reward_function = config.get('initial_reward_function', None)

        if self.initial_reward_function is not None:
            self._prepare_initial_reward_function()


        self.logging(str(config), logging.INFO)

        os.makedirs(self.reward_function_path, exist_ok=True)
        if self.api_key is not None:
            openai.api_key = self.api_key
        else:
            try:
                openai.api_key = file_to_string(path.join(path.expanduser("~"), "API_key.txt"))
            except:
                self.logging(logging.CRITICAL, "API key not found. Please provide the API key using 'api_key' in config")
                raise Exception("Raise Error")



    def logging(self, message, level=logging.DEBUG):
        info_dict = {
            'outer_loop': self.iteration_num if hasattr(self, 'iteration_num') else -1,
            'inner_loop': self.current_inner,
            'n_inner': self.n_inner,
            'trial': self._current_trial if hasattr(self, '_current_trial') else -1,  # Assuming self._current_trial is a class attribute
            'trial_count': self.trial_count if hasattr(self, 'trial_count') else -1,
        }

        # Define the prefix format
        prefix = '[ol: {outer_loop}, il: {inner_loop} / {n_inner}, trial: {trial} / {trial_count}]'.format(**info_dict)

        # Split the message by line breaks and log each line with the prefix
        message = str(message)
        for line in message.splitlines():
            formatted_message = f'{prefix} {line}'
            logging.log(level, formatted_message)

    def _prepare_initial_reward_function(self):
        # Copy the initial reward function to the reward function path
        reward_file_name = basename(self.initial_reward_function)

        initial_reward_function_path = path.join(self.reward_function_path, reward_file_name)
        # Copy
        shutil.copy(self.initial_reward_function, initial_reward_function_path)

        self.generating_function_path = initial_reward_function_path
        self.previous_reward_function = file_to_string(self.generating_function_path)

        self.logging(f"Copied the initial reward function to the reward function path: {self.initial_reward_function} -> {initial_reward_function_path}", logging.INFO)

    def start_chat(self, model, messages, max_tokens, log_dict=None, log_key='first_user', passthrough_response=None,
                   verbose=False, seed=42):
        try:
            if passthrough_response is None:
                if verbose:
                    self.logging("Sending the request: ", messages)

                client = UnifiedLLMClient()
                ctx = ChatContext()

                responses = client.call_model(ctx, messages, model=model, seed=seed, n_response=1)
                response = responses[0][0]
                context = responses[0][1]

                if verbose:
                    self.logging("Received the response: ", response)
            else:
                try:
                    response = file_to_string(passthrough_response)
                except FileNotFoundError as e:
                    self.logging(logging.CRITICAL, "File not found: {passthrough_response}\n", e)
                    raise Exception("Raise Error")

            if log_dict is not None:
                log_dict[log_key] = dict(request=messages, response=response)

        except KeyboardInterrupt:
            raise Exception("Keyboard Interrupt while using the OpenAI API")

        return response, context

    def run(self):

        while self.current_inner <= self.n_inner:
            reward_function_name = f"{self.postfix}_inner_{self.current_inner}"
            is_success = False

            generating_function_path = None
            generating_function_error = None

            if hasattr(self, 'generating_function_path'):
                generating_function_path = self.generating_function_path
                self.previous_reward_function_path = self.generating_function_path
                del self.generating_function_path
                self.logging(f"Using the initial reward function: {generating_function_path}", logging.INFO)

            self.logging(f"Generating reward function: {generating_function_path}", logging.DEBUG)

            for i_trial in range(1, self.trial_count + 1):

                self._current_trial = i_trial
                basename = f"{reward_function_name}_trial_{i_trial}"

                self.logging(f"Generating reward function: {basename}", logging.INFO)

                if self.current_inner == 1 and self.iteration_num == 1:
                    self.logging(f'Calling the zero-shot generation function. (len(error): {len(generating_function_error) if generating_function_error else 0})')
                    generating_function_path = self.first_user_response(basename=basename,
                                                                        generating_function_path=generating_function_path,
                                                                        generating_function_error=generating_function_error,
                                                                        trial=i_trial)
                    self.logging(f'Called the first_user_response function')
                else:
                    self.logging(f'Calling the inner-loop generation function. (len(error): {len(generating_function_error) if generating_function_error else 0})')
                    generating_function_path = self.second_user_response(basename=basename,
                                                                        generating_function_path=generating_function_path,
                                                                        generating_function_error=generating_function_error,
                                                                        trial=i_trial)
                    self.logging(f'Called the second_user_response function')

                execute_output, execute_msg = self.execute_reward_function(generating_function_path)

                if is_convertible_to_float(execute_output):
                    self.previous_reward_function = file_to_string(generating_function_path)
                    self.previous_reward_function_path = generating_function_path
                    is_success = True
                    break
                else:
                    error_message = execute_msg
                    generating_function_error = error_message

            if not is_success:
                raise Exception(f"Failed to generate the reward function. Please check the error message.\n{error_message}")

            self.current_inner += 1

        # Save the reward function to the file
        reward_function_string = file_to_string(generating_function_path)
        reward_function_file_path = path.join(self.reward_function_path, f"{reward_function_name}.py")
        with open(reward_function_file_path, 'w') as f:
            f.write(reward_function_string)

        print("Done")

    def first_user_response(self, basename: str = 'reward', generating_function_path: str = None, generating_function_error: str = None, trial=1):

        self.initial_system = self.initial_system.format(
            i='{i}',
            reward_signature=self.reward_template,
            sampled_data_example=self.sampled_data_example
        )

        initial_user = copy.deepcopy(self.initial_user)

        if generating_function_path is not None and generating_function_error is not None:

            reward_code = file_to_string(generating_function_path)

            sample_code = """
            ## Previous Reward Code
            Here is the previous reward function that you have generated. However, this code has an error. Please fix the error and generate the reward function again.
            ```python
            {reward_code_string}
            ```
            Error Message:
            {error_message}
            
            """.format(reward_code_string=reward_code, error_message=generating_function_error)

            initial_user = initial_user.format(
                few_shot_code_string=sample_code
            )
        else:
            sample_code = """
            ## Example Reward Code
            Here is the example of the reward function which minimizes the error between target (State.Target.WinRate) and current win rate (Playtesting.WinRate).
            The function measure the decrease/increase of the error by comparing the previous and current winrate error.
            ```python
            {task_obs_code_string}
            ```
            """.format(task_obs_code_string=self.reward_example)

            initial_user = initial_user.format(
                few_shot_code_string=sample_code
            )

        messages = [
            {"role": "system", "content": self.initial_system},
            {"role": "user", "content": initial_user}
        ]

        # print(messages)
        response, context = self.start_chat(self.gpt_model, messages, self.gpt_max_token, seed=trial)
        self.logging(context, logging.INFO)
        self.logging(response, logging.DEBUG)

        response_file_path = path.join(self.reward_function_path, f"{basename}.response.pkl")
        with open(response_file_path, 'wb') as f:
            pickle.dump(response, f)

        context_file_path = path.join(self.reward_function_path, f"{basename}.context.pkl")
        with open(context_file_path, 'wb') as f:
            pickle.dump(context, f)

        # response_prompt = response.choices[0].message.content
        parsed_reward_function = parse_reward_function(response)

        log_dict = {
            'request': messages,
            'response': response,
        }

        # Save reward function to .py
        reward_file_path = path.join(self.reward_function_path, f"{basename}.py")
        with open(reward_file_path, 'w') as f:
            f.write(parsed_reward_function)

        # Save the log to .json file
        log_file_path = path.join(self.reward_function_path, f"{basename}.json")
        with open(log_file_path, 'w') as f:
            json.dump(log_dict, f, indent=4)

        # if the preprocessing code do not exists in the reward path, copy it to the past
        self._ensure_utility_files(self.reward_function_path)

        return reward_file_path

    def second_user_response(self, basename: str = 'reward', generating_function_path: str = None, generating_function_error: str = None, trial=1):
        playtesting_result = ""
        parsed_code = ast.parse(self.previous_reward_function)
        error_message = None

        sampled_data_path = abspath(path.join(self.reward_function_path, f"{basename}.data.json"))
        preprocess_dataset(self.arbitrary_dataset, sampled_data_path)
        # read the sampled data and save the lines with array

        # This section is for module test of the reward function

        sample_data_arr = list()
        with open(sampled_data_path, 'r') as file:
            for line in file:
                sample_data_arr.append(preprocessing(json.loads(line)))

        try:
            for node in ast.walk(parsed_code):
                if isinstance(node, ast.FunctionDef) and node.name == 'compute_reward':

                    inner_node_list = list()

                    for inner_node in ast.iter_child_nodes(node):
                        if isinstance(inner_node, ast.FunctionDef):
                            inner_node_list.append(inner_node.name)

                            nested_function_code = astor.to_source(inner_node)
                            exec(nested_function_code, globals())

                    self.logging(f'Founded ({len(inner_node_list)}) sub reward functions: {inner_node_list} and sample data: {len(sample_data_arr)}', logging.DEBUG)

                    result_dict = dict()

                    for inner_node in ast.iter_child_nodes(node):
                        if isinstance(inner_node, ast.Assign):
                            if isinstance(inner_node.value, ast.Constant):
                                exec(astor.to_source(inner_node), globals())

                        if isinstance(inner_node, ast.FunctionDef):

                            try:
                                result_list = list()

                                # Input the arbitrary game data to the sub reward function
                                for kwarg in sample_data_arr:
                                    result = globals()[inner_node.name](kwarg['Current'])
                                    result_list.append(result)

                                average_value = sum(result_list) / len(result_list)
                                standard_deviation = sum([(x - average_value) ** 2 for x in result_list]) / len(
                                    result_list) ** 0.5

                                result_dict[inner_node.name] = {'Average': average_value, 'Standard deviation': standard_deviation}

                            except:
                                error_msg = traceback.format_exc()
                                result_dict[inner_node.name] = {'Error': error_msg }
                                self.logging(error_msg, logging.ERROR)

                    # End of loop

            # Result for sub reward functions
            if len(result_dict) > 0:
                _playtesting_result = "[Sub-reward Output Analysis]\n"

                for node_name, item in result_dict.items():
                    if isinstance(item, dict):
                        _playtesting_result += f"({node_name}) "  # node_name 한 번만 출력
                        result_line = []
                        for key, value in item.items():
                            # 실수는 소수점 3자리까지 출력
                            if isinstance(value, float):
                                value_str = f"{value:.3f}"
                            else:
                                value_str = str(value)

                            # key-value 쌍을 리스트에 추가
                            result_line.append(f"{key}: {value_str}")

                        # 각 node_name에 대한 결과를 한 줄로 출력
                        _playtesting_result += '  '.join(result_line) + '\n'


                playtesting_result =_playtesting_result.strip()  # 마지막 공백 제거

            # Result for the main reward function
            required_node_list = ['agent_0', 'agent_1', 'agent_2', 'agent_3']
            # 부족한 항목과 추가된 항목을 비교
            missing_nodes = set(required_node_list) - set(inner_node_list)
            extra_nodes = set(inner_node_list) - set(required_node_list)

            # 메시지 작성
            node_message = ""

            if extra_nodes:
                node_message += f"\nExtra nodes found, please remove: {', '.join(extra_nodes)}\n"
            if missing_nodes:
                node_message += f"\nMissing nodes: {', '.join(missing_nodes)}\n"

            playtesting_result += node_message

        except:
            self.logging(traceback.format_exc(), logging.ERROR)
            error_message = traceback.format_exc()

        # End of the module t est of the reward function

        # Start of the execution test of the reward function
        reward_mean, reward_std, success_rate = self.execute_reward_functions_parallel(self.previous_reward_function_path, state_dicts=sample_data_arr)
        playtesting_result += '\n[Total Reward Analysis]\n'
        playtesting_result += 'Average: {:.3f}\n'.format(reward_mean)
        playtesting_result += 'Standard deviation: {:.3f}\n'.format(reward_std)
        playtesting_result += 'Success Rate: {:.1f}%\n'.format(success_rate)

        self.logging(playtesting_result, logging.DEBUG)

        if generating_function_error is not None:
            error_description = " The previous reward function has an error. Below is the error message. Please generate the reward function again with attention to error.\n" + generating_function_error
        elif error_message is not None:
            error_description = " The previous reward function has an error. Below is the error message. Please generate the reward function again with attention to error.\n" + error_message
        else:
            error_description = ""

        self.second_user = file_to_string(path.join(self.file_path, "second_user.txt"))
        self.second_user = self.second_user.format(
            i='{i}',
            previous_reward_function=self.previous_reward_function,
            error_description=error_description,
            playtesting_result=playtesting_result
        )
        messages = [
            {"role": "system", "content": self.initial_system},
            {"role": "user", "content": self.second_user}
        ]

        response, context = self.start_chat(self.gpt_model, messages, self.gpt_max_token, seed=trial)
        self.logging(context, logging.INFO)
        self.logging(response, logging.DEBUG)

        os.makedirs(self.reward_function_path, exist_ok=True)
        response_file_path = path.join(self.reward_function_path, f"{basename}.response.pkl")
        with open(response_file_path, 'wb') as f:
            pickle.dump(response, f)

        context_file_path = path.join(self.reward_function_path, f"{basename}.context.pkl")
        with open(context_file_path, 'wb') as f:
            pickle.dump(context, f)

        parsed_reward_function = parse_reward_function(response)

        log_dict = {
            'request': messages,
            'response': response,
        }

        # Save reward function to .py
        reward_file_path = path.join(self.reward_function_path, f"{basename}.py")
        with open(reward_file_path, 'w') as f:
            f.write(parsed_reward_function)

        # Save the log to .json file
        log_file_path = path.join(self.reward_function_path, f"{basename}.json")
        with open(log_file_path, 'w') as f:
            json.dump(log_dict, f, indent=4)

        return reward_file_path

    def _ensure_utility_files(self, path: str) -> None:
        # Get the directory where the script is located
        original_dir = os.path.dirname(__file__)

        # Define the utility directory within the original directory (assuming it's called 'utils')
        utility_dir = os.path.join(original_dir, 'example', 'utils')

        # Copy all files from utility_dir to the destination path
        for filename in os.listdir(utility_dir):
            file_path = os.path.join(utility_dir, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, path)

        return None

    def execute_reward_function(self, reward_function_path: str, current_state_path: str = None) -> (str, str):
        """
        보상 함수를 실행하는 함수. current_state_path를 인자로 받아 임시 경로를 사용할 수 있으며,
        기본값으로 self.current_state_path를 사용한다.
        """

        # current_state_path가 None이면 self.current_state_path 사용
        if current_state_path is None:
            current_state_path = self.current_state_path

        # Build the command line to execute the reward function
        code_execution_command_line = ['python', abspath(reward_function_path), current_state_path]

        # Logging the command that will be executed
        self.logging(f"Executing the reward function with the command line: {' '.join(code_execution_command_line)}", logging.INFO)

        # Execute the reward function and capture output and error
        process = subprocess.run(code_execution_command_line, capture_output=True, text=True)

        # Capture stdout and stderr
        output = process.stdout
        error = process.stderr

        if output == '' and error == '':
            error = 'no output'

        # Logging the result of the execution
        if error:
            self.logging(f"Error occurred while executing the reward function (len: {len(error)}) - error:\n{error}", logging.WARNING)
        else:
            self.logging(f"Executed the reward function - result: {output}", logging.INFO)

        # Return the output and error
        return output, error

    def execute_reward_function(self, reward_function_path: str, current_state_path: str = None, verbose: bool = False) -> (str, str):
        """
        보상 함수를 실행하는 함수. current_state_path를 인자로 받아 임시 경로를 사용할 수 있으며,
        기본값으로 self.current_state_path를 사용한다.
        """

        # current_state_path가 None이면 self.current_state_path 사용
        if current_state_path is None:
            current_state_path = self.current_state_path

        # Build the command line to execute the reward function
        code_execution_command_line = ['python', abspath(reward_function_path), current_state_path]

        # Logging the command that will be executed
        if verbose:
            self.logging(f"Executing the reward function with the command line: {' '.join(code_execution_command_line)}", logging.INFO)

        # Execute the reward function and capture output and error
        process = subprocess.run(code_execution_command_line, capture_output=True, text=True)

        # Capture stdout and stderr
        output = process.stdout
        error = process.stderr

        if output == '' and error == '':
            error = 'no output'

        # Logging the result of the execution
        if error:
            if verbose:
                self.logging(f"Error occurred while executing the reward function (len: {len(error)}) - error:\n{error}", logging.WARNING)
        else:
            if verbose:
                self.logging(f"Executed the reward function - result: {output}", logging.INFO)

        # Return the output and error
        return output, error

    def execute_single(self, reward_function_path: str, state_data: dict, verbose: bool = True) -> (str, str):
        """
        개별 보상 함수를 실행하는 메서드. state_data(dict)를 임시 파일로 저장하고,
        그 파일의 경로를 보상 함수에 넘겨준다.
        """
        # 임시 파일에 state_data 저장
        with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
            json.dump(state_data, temp_file)
            temp_file_path = temp_file.name

        # 보상 함수 실행
        result = self.execute_reward_function(reward_function_path, temp_file_path, verbose)

        # 임시 파일 삭제
        try:
            os.remove(temp_file_path)
        except OSError as e:
            self.logging(f"Error deleting temp file: {e}", logging.ERROR)

        return result

    def execute_reward_functions_parallel(self, reward_function_path: str, state_dicts: list) -> list:
        """
        여러 개의 보상 함수를 병렬로 실행하는 함수.
        state_dicts는 각 보상 함수에서 사용할 상태 데이터 딕셔너리들의 리스트다.
        """

        self.logging(f"Executing reward functions in parallel: {len(state_dicts)} state dicts", logging.INFO)

        # 병렬 처리를 위한 프로세스 풀 생성
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # 각 상태 데이터를 임시 파일로 저장하고 병렬로 작업 실행
            results = pool.starmap(self.execute_single, [(reward_function_path, state_data, False) for state_data in state_dicts])

        self.logging(f"Executed reward functions in parallel: {len(state_dicts)} state dicts", logging.INFO)

        reward_values = list()
        error_messages = list()
        for result, error in results:
            if is_convertible_to_float(result):
                reward_values.append(float(result))
            else:
                error_messages.append(error)

        # Get the mean and std
        mean_reward = np.mean(reward_values)
        std_reward = np.std(reward_values)

        num_inputs = len(state_dicts)
        num_success = len(reward_values)
        success_rate = num_success / num_inputs * 100

        # 모든 결과 반환
        return mean_reward, std_reward, success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--shared_storage_path', type=str, default='.')
    parser.add_argument('--postfix', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    parser.add_argument('--reward_functions_dir', type=str, default='RewardFunctions')
    parser.add_argument('--gpt_model', type=str, default='llama3-70b-instruct')
    parser.add_argument('--gpt_max_token', type=int, default=4096)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--current_inner', type=int, default=1)
    parser.add_argument('--n_inner', type=int, default=1)
    parser.add_argument('--n_outer', type=int, default=1)
    parser.add_argument('--reference_csv', type=str, default='random_dataset.txt')
    parser.add_argument('--arbitrary_dataset', type=str, default='./example/random_dataset.txt')
    parser.add_argument('--trial_count', type=int, default=10)
    parser.add_argument('--iteration_num', type=int, default=1)
    parser.add_argument('--previous_reward_function', type=str, default='compute_reward_example.py')

    parser.add_argument('--initial_reward_function', type=str, default=None)

    args = parser.parse_args()

    args = vars(args)
    reward_generator = RewardGenerator(args)
    reward_generator.run()