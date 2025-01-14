import os
import random
import json
import numpy as np

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_judge_prompts(prompt_file: str):
    """Load judge prompts.

    The return value is a python dict of type:
    Dict[judge_name: str -> dict]
    """
    prompts = {}
    with open(prompt_file, 'r', encoding="utf-8") as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


PROMPT_TEMPLATES = {
    "qwen25": (
        "<|im_start|>system\n{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "gemma2": (
        "<bos><start_of_turn>user\n"
        "{system_prompt}\n\n"
        "{user_prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    ),
    "llama3": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
}