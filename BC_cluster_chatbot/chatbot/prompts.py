import yaml
import os
from enum import Enum

absolute_path = os.path.abspath(__file__)
directory = os.path.dirname(absolute_path)
prompts_file = f"{directory}/prompts.yaml"

with open(prompts_file, 'r') as file:
    prompts = yaml.safe_load(file)


class Prompts(Enum):
    master='MASTER'
    rag='RAG'
    predict='PREDICT'

    def get_prompt(self):
        return prompts[f"{self.value}_PROMPT"]