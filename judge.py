import os
import time
import json
import argparse
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import set_seed, load_judge_prompts, PROMPT_TEMPLATES, save_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--prompt_path", default="./data/judge_prompts.jsonl", type=str)
    parser.add_argument("--model_name_or_path", default="Qwen2.5-1.5B-Instruct", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="pair-v2", type=str)
    parser.add_argument("--num_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--switch_order", action="store_true")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    args = parser.parse_args()
    # top_p must be 1 when using greedy sampling (vllm)
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


def setup(args):
    # Load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
    )
    tokenizer = None
    if args.apply_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )

    # Infer & eval
    main(llm, tokenizer, args)


def prepare_data(args):
    df = pd.read_parquet(args.data_path)

    # sample `num_test_sample` from dataset
    if args.num_sample > 0:
        df = df[: args.num_sample]

    # select start and end
    df = df[args.start : len(df) if args.end == -1 else args.end]

    # get out_file name
    out_file_prefix = f"switch{args.switch_order}_{args.prompt_type}_{args.num_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}", exist_ok=True)

    # Process
    examples = []
    for index, row in df.iterrows():
        if row["prompt"] is not None and row["winner"] is not None \
           and row["response_a"] is not None and row["response_b"] is not None:
            if args.switch_order:
                maps = {"model_a": "model_b", "model_b": "model_a"}
                example = {
                    "idx": index,
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "response_a": row["response_b"],
                    "response_b": row["response_a"],
                    "winner": maps[row["winner"]],
                    "model_a": row["model_b"],
                    "model_b": row["model_a"],
                    "language": row["language"],
                }
            else:
                example = {
                    "idx": index,
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "response_a": row["response_a"],
                    "response_b": row["response_b"],
                    "winner": row["winner"],
                    "model_a": row["model_a"],
                    "model_b": row["model_b"],
                    "language": row["language"],
                }
            examples.append(example)
        else:
            print(f"None in row {index}")
    return examples, out_file


def main(llm, tokenizer, args):
    # Prepare data
    examples, out_file = prepare_data(args)

    # Prepare prompts
    judge_prompt = load_judge_prompts(args.prompt_path)["pair-v2"]
    system_prompt = judge_prompt["system_prompt"]
    input_template = PROMPT_TEMPLATES[args.prompt_type]

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]
        user_prompt = judge_prompt["prompt_template"].format(
            question=example["prompt"],
            answer_a=example["response_a"],
            answer_b=example["response_b"],
        )
        full_prompt = input_template.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["prompt"],
            "prompt": full_prompt,
            "response_a": example["response_a"],
            "response_b": example["response_b"],
            "winner": example["winner"],
            "model_a": example["model_a"],
            "model_b": example["model_b"],
            "language": example["language"],
        }
        samples.append(sample)

    # start inference
    if args.prompt_type == "qwen25":
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    elif args.prompt_type == "gemma2":
        stop_words = ["<eos>", "<end_of_turn>"]
        
    start_time = time.time()
    input_prompts = [(i, sample["prompt"]) for i, sample in enumerate(samples)]
    prompts = [item[1] for item in input_prompts]
    outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                    stop_token_ids=(
                        [151645, 151643]
                        if "qwen2" in args.model_name_or_path.lower()
                        else None
                    ),
                ),
            )
    outputs = sorted(outputs, key=lambda x: int(x.request_id))  # sort outputs by request_id
    outputs = [output.outputs[0].text for output in outputs]
    assert len(outputs) == len(input_prompts)

    # process all outputs
    completions = []
    preds = []
    for output in outputs:
        output = output.rstrip()
        completions.append(output)

        if "[[A]]" in output:
            winner = "model_a"
        elif "[[B]]" in output:
            winner = "model_b"
        else:
            winner = "error"
        preds.append(winner)

    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        winner = sample["winner"]
        sample.pop("prompt")
        sample.pop("winner")
        sample.update({"completion": completions[i], "pred": preds[i], "winner": winner})
        all_samples.append(sample)

    # save outputs
    if args.save_outputs:
        save_jsonl(all_samples, out_file)

    # evaluate
    cnt_correct = 0
    total = 0
    for i, sample in enumerate(all_samples):
        total += 1
        if sample["pred"] == sample["winner"]:
            cnt_correct += 1
    accuracy = cnt_correct / total
    print(f"Accuracy: {accuracy}, num_samples: {len(all_samples)}")

    # save metrics
    result_json = {
        "num_samples": len(all_samples),
        "time_use_in_second": time_use,
        "time_use_in_minute": f"{int(time_use // 60)}:{int(time_use % 60):02d}",
        "accuracy": accuracy
    }
    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)