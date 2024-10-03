import ast
import re
import pandas as pd
from os import path
import json

def file_to_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_function_signature(code_string, time):
    # Parse the code string into an AST
    module = ast.parse(code_string)

    # Find the function definitions
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]

    # If there are no function definitions, return None
    if not function_defs:
        return None

    # For simplicity, we'll just return the signature of the first function definition
    function_def = function_defs[0]

    input_lst = []
    # Construct the function signature (within object class)
    signature = function_def.name + '(self.' + ', self.'.join(arg.arg for arg in function_def.args.args) + ')'
    for arg in function_def.args.args:
        input_lst.append(arg.arg)
    return signature, input_lst


def parse_reward_function(message):
    patterns = [
        r'```python(.*?)```',
        r'```(.*?)```',
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]

    for pattern in patterns:
        code_string = re.search(pattern, message, re.DOTALL)
        if code_string is not None:
            code_string = code_string.group(1).strip()
            break
    code_string = message if not code_string else code_string

    return code_string


def filter_dataframe(df: pd.DataFrame, iteration: list, return_columns: list):
    df = df[df['LLM.RewardFunctionIteration'].isin(iteration)]
    df = df[return_columns]
    return df


def is_convertible_to_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def min_max_normalize(df):
    return (df - df.min()) / (df.max() - df.min())


def preprocess_dataset(skill_log_path, sampled_data_path):
    data = []
    with open(skill_log_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    skill_log_df = pd.DataFrame(data)
    skill_log_df = skill_log_df.sample(n=20)
    skill_log_df.to_json(sampled_data_path, orient='records', lines=True)

    return None