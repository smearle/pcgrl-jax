import os
import time
import json
import copy
from openai import OpenAI

temperature = 0
max_time = 99999
token_limit = 25000

class ChatContext:
    def __init__(self):

        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.response_times = []
        self.chats = []

    def add_interaction(self, user_message, assistant_message, input_tokens, output_tokens, response_time, model_name):
        self.total_tokens += input_tokens + output_tokens
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.response_times.append(response_time)
        self.chats.append({
            "user_message": user_message,
            "assistant_message": assistant_message,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response_time": response_time,
            "model_name": model_name
        })

    def __repr__(self):
        return (
            f"ChatContext(total_tokens={self.total_tokens}, input_tokens={self.input_tokens}, "
            f"output_tokens={self.output_tokens}, response_times={self.response_times}, len_chats={len(self.chats)})")

    def to_json(self):
        return {
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "response_times": self.response_times,
            "chats": self.chats
        }


class UnifiedLLMClient:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')

        self._config_path = config_path
        self._read_config()

    def _read_config(self):
        with open(self._config_path, 'r') as f:
            self.config = json.load(f)

    def call_model(self, ctx: ChatContext, messages, model=None, n_response=1, seed=42, top_p=0.99):
        model_config = self.config[model]
        api_host = model_config["api_host"]
        api_key = model_config["api_key"]
        full_model_name = model_config["full-model-name"]
        extra_body = model_config.get("extra_body", {})

        client = OpenAI(timeout=max_time, base_url=api_host, api_key=api_key)

        start_time = time.time()

        response = client.chat.completions.create(
            model=full_model_name,
            messages=messages,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            max_tokens=4096,
            n=n_response,
            extra_body=extra_body
        )

        end_time = time.time()

        responses = list()
        for choice in response.choices:
            response_time = end_time - start_time
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            assistant_message = choice.message.content

            _ctx = copy.deepcopy(ctx)
            _ctx.add_interaction(
                user_message=messages[-1]['content'],
                assistant_message=assistant_message,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_time=response_time,
                model_name=model
            )
            responses.append((assistant_message, _ctx))

        return responses


if __name__ == "__main__":


    # Set the model name

    client = UnifiedLLMClient()
    ctx = ChatContext()

    # Call the model
    messages = [
        {
            "role": "system",
            "content": "# PCG Agent Reward Function Generation Task\nYou are a reward function engineer trying to write reward functions to solve reinforcement learning tasks as effective as possible.\nThe PCG agent is an agent that balances the game environment by adjusting the setting of the game variables related to the game difficulty.\nThe adjustable variables are health, armor, and speed of the player agents and range, cooldown, and damage of the players's attack skill.\nThe action of the PCG agent revise the player property value, which is one of four players, to balance the game difficulty and the reward function evaluates the game difficulty based on the playtested results.\nThe state of the agent is current game setting values and the action is adjustment of the game setting values.\nOn every episode, the game setting values are initialized randomly and the PCG agent adjusts the game setting values to achieve the target win rate.\nThe agent gets the reward signal from the reward function you write and learns to adjust the game setting values to achieve the goal of the reward function.\n\n## The Raid Game Environment\nThe game environment is a multiplayer game where player agents fight against a boss agent (i.e., boss raid game).\nThere are four ally player agents and one boss agent in the game and the goal of the player agents is to defeat the boss agent.\nOn the beginning of the simulation, the game setting values are deployed to the game environment and the player agents and the boss agent are spawned at random locations on the map.\nNext, the environment starts game and repeats the game by an arbitrary number (e.g., 100) of episodes to simulate the deployed game settings.\nOn the end of the simulation, the environment collects, calculates, and store the playtested results via an output file.\n\n## Variable Reference\nIn this section, the variables that the reward function can access are described.\nThe reward function only can access the key listed below. If the key is not listed below, the reward function cannot access the value.\nThe common variables are the variables that are measured for the overall game state, not for each player.\nThe individual variables are the variables that are measured for each player in the game state.\n\n## Individual Variables\n`Playtesting.Agent{i}.SurviveTime` - The survival time of Agent {i} during playtesting.\n`Playtesting.Agent{i}.Distance.Moved.PerSecond` - The average distance moved per second by Agent {i}.\n`Playtesting.Agent{i}.Distance.Boss.Mean` - The average distance of Agent {i} from the boss entity.\n`Playtesting.Agent{i}.Damage.Dealt.PerSecond` - The average damage dealt per second by Agent {i}.\n`Playtesting.Agent{i}.Damage.Taken.PerSecond` - The average damage taken per second by Agent {i}.\n`Playtesting.Agent{i}.Armored.PerSecond` - The change in armor status per second for Agent {i}.\n`Playtesting.Agent{i}.Health.Last.Ratio` - The ratio of Agent {i}'s last health value to its maximum health.\n`Playtesting.Agent{i}.Skill.Used.PerSecond` - The average usage of the specific skill per second by Agent {i}.\n\nThe playtested values are min-max normalized for each variable. The values are normalized to the range of [0, 1].\nThere are four player agents in the game and the index of the player agent is from 0 to 3. (e.g., Agent0, Agent1, Agent2, Agent3)\nThe example of the key name is \"Playtesting.Agent0.SurviveTime\" for the survival time of Agent0.\n\n## Reward Function\nThe reward function is a function that calculates the reward value for the agent based on the playtested results.\nThe function is written in Python and loads the playtested results from the json file and calculates the reward value based on the results.\n\nimport json\nimport sys\nimport numpy as np\n\n\ndef compute_reward(kwarg):\n    reward = 0.0\n\n    # start of code\n    def reward_1(kwarg):\n        return 0.0\n\n    def reward_2(kwarg):\n        return 0.0\n\n    def reward_3(kwarg):\n        return 0.0\n\n\n    reward = reward_1(kwarg) + reward_2(kwarg) + reward_3(kwarg)\n    # end of code\n\n    return reward\n\n\n# Do not edit this part (start)\nif __name__ == \"__main__\":\n    try:\n        json_path = sys.argv[1]\n\n        with open(json_path, 'r') as f:\n            kwarg = json.load(f)\n\n        reward = compute_reward(kwarg['Current'])\n        print(reward)\n    except IndexError:\n        print(\"Error: No argument provided.\")\n# Do not edit this part (end)\n\nThis is the template of the reward function.\nThe 'compute_reward' function is composed by summing the results from multiple reward terms, such as reward_1, reward_2 ...\nSimilar to the template provided, it is necessary to create functions within the function, and the number of functions does not matter.\nThe function receives the playtested results and returns the reward value in float.\nThe function should be implemented in the \"compute_reward\" function.\nThe reward shaping code should be written between '# start of code' and '# end of code' comments.\nThe code output should be formatted as a Python code string: \"```python ... ```\"."
        },
        {
            "role": "user",
            "content": "# The Task\nYour task is to generate a reward function for the PCG agent which works in the Raid environment.\nIn order to maximize the fun of multiplayer games, it is possible to express different skills for the four players being generated.\nThe goal is to find insights that can diversify the parameter of four player agents and write code that measure how the skill and stats of the four agents (Agent0, 1, 2, and 3) clearly distinct.\nNote that the PCG agent revises the game setting of one of the player agents in round-robin manner.\nAccordingly, the reward function should evaluate the playtesting result and compare the improvement with previous result.\n\nYou can design factors to generate the reward function, and properly sum them up to get the final reward.\nUtilize values of the playtesting results on the implementation of the design factors.\nFor stability of learning, design the reward to be returned in the range [0,1].\n\n\n            ## Example Reward Code\n            Here is the example of the reward function which minimizes the error between target (State.Target.WinRate) and current win rate (Playtesting.WinRate).\n            The function measure the decrease/increase of the error by comparing the previous and current winrate error.\n            ```python\n            import json\nimport sys\nimport numpy as np\n\n\ndef compute_reward(kwarg):\n    reward = 0.0\n\n    # start of code\n\n    def reward_1(kwarg) -> float:\n        # Dictionary usage example\n        # kwarg['State.Agent0.Property.Health.Max']\n        # kwarg['Playtesting.Agent1.Skill0.Used.PerSecond']\n\n        return 0.0 # Return the float value\n\n    def reward_2(kwarg) -> float:\n        # Dictionary usage example\n        # kwarg['State.Agent0.Property.Health.Max']\n        # kwarg['Playtesting.Agent1.Skill0.Used.PerSecond']\n\n        return 0.0 # Return the float value\n\n    diversity = reward_1(kwarg) + reward_2(kwarg)\n\n    # Calculate the improvement\n    reward += diversity\n\n    # end of code\n\n    return reward\n\n\nif __name__ == \"__main__\":\n    try:\n        json_path = sys.argv[1]\n\n        with open(json_path, 'r') as f:\n            kwarg = json.load(f)\n\n        reward = compute_reward(kwarg['Current'])\n        print(reward)\n    except IndexError:\n        print(\"Error: No argument provided.\")\n            ```\n            \n\n\nFind insight(s) to design the reward function and write it in the Python code.\nDo not change the form of reward function and the argument of nested function .\n\nSome helpful tips for writing the reward function code:\n(1) You may find it helpful to normalize the reward to a fixed range by applying transformations like torch.exp to the overall reward or its components\n(2) If you choose to transform a reward component, then you must also introduce a temperature parameter inside the transformation function; this parameter must be a named variable in the reward function and it must not be an input variable. Each transformed reward component should have its own temperature variable\n(3) Do not write comments in the code.\n(4) Most importantly, the reward code's input variables must contain only attributes of the provided environment class definition (namely, variables that have prefix self.). Under no circumstance can you introduce new input variables.\n(5) All nested functions must accept only one argument, 'kwarg'. They must not accept any other arguments under any circumstances.\n(6) In the reward function you creats, do not change the form of 'reward = compute_reward(kwarg['Current'])' inside the 'if __name__ == \"__main__\"' block. Ensure that only the argument value of 'kwarg['Current']' is used.\n\n<INSIGHTS>\n-\n-\n</INSIGHTS>\n\nReward function:\n```python\n<CODE>\n</CODE>\n```"
        }
    ]

    responses = client.call_model(ctx, messages, model='llama3-70b-instruct', n_response=1)
    #
    print(f"Response:", responses[0][1])

    ctx = ChatContext()

    responses = client.call_model(ctx, messages, model='llama3-70b-instruct', n_response=1)

    print(f"Response:", responses[0][1])

