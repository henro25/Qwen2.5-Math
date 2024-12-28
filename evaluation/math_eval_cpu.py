import random
import os
import argparse
import time
import json
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Removed imports for vllm, torch, transformers, etc.
# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer, AutoModelForCausalLM

# Keep local imports
from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import parse_question, parse_ground_truth, run_execute
from trajectory import extract_program
from data_loader import load_data
from python_executor import PythonExecutor
# Removed: from model_utils import load_hf_lm_and_tokenizer, generate_completions

# Import your inference endpoint client
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="llama-3-2-1b-instruct", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt (if relevant).",
    )
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    # Additional arguments to store API base/key
    parser.add_argument("--openai_api_key", default="EMPTY", type=str)
    parser.add_argument(
        "--openai_api_base",
        default="https://ray-stable-killdeer.ngrok-free.app/v1",
        type=str,
    )
    args = parser.parse_args()

    # top_p must be 1 when using temperature=0 (greedy sampling style)
    if args.temperature == 0:
        args.top_p = 1

    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : (len(examples) if args.end == -1 else args.end)]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = (
        f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    )
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # deduplicate
    processed_samples_dict = {}
    for sample in processed_samples:
        processed_samples_dict[sample["idx"]] = sample

    processed_idxs = list(processed_samples_dict.keys())
    processed_samples = list(processed_samples_dict.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup_inference_client(args):
    """
    Initialize the OpenAI client for your custom inference endpoint.
    """
    # Create your endpoint client
    client = OpenAI(
        api_key=args.openai_api_key,
        base_url=args.openai_api_base,
    )
    return client


def generate_completions_inferencing_endpoint(client, prompts, args):
    """
    Given a list of input prompts, call the inference endpoint for each one
    and return a list of generated outputs.
    """
    
    # This list will hold the final outputs in the correct order
    outputs = [None] * len(prompts)
    
    def run_inference(index, prompt):
        """Worker function that calls the endpoint for a single prompt."""
        messages = [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model=args.model_name_or_path,
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            n=1,
            # If your endpoint supports stop sequences, you can add them here:
            # stop=stop_words
        )
        text_output = response.choices[0].message.content.strip()
        return index, text_output

    # You can tune the number of worker threads if you want
    max_workers = 100  # Example: 5 threads

    # Submit tasks
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Initialize progress bar
        with tqdm(total=len(prompts), desc="Inferencing") as progress_bar:
            for idx, prompt in enumerate(prompts):
                # Submit a job to the thread pool
                futures.append(executor.submit(run_inference, idx, prompt))

            # Iterate over the futures as they complete
            for future in as_completed(futures):
                index, result = future.result()
                outputs[index] = result
                progress_bar.update(1)  # Update the progress bar after each result

    return outputs

def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(client, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]

    # If you have an "apply_chat_template", you can do it here if needed
    if args.apply_chat_template:
        # Example: just a trivial transform; adjust as needed
        input_prompts = [("System: You are a helpful assistant.\n" + p) for p in input_prompts]

    remain_prompts = [(i, prompt) for i, prompt in enumerate(input_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    # You can define your own stop words if the endpoint supports them
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs via your inference endpoint
        prompts = [item[1] for item in current_prompts]
        outputs = generate_completions_inferencing_endpoint(client, prompts, args)
        assert len(outputs) == len(current_prompts)

        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            # Determine if we need another pass or if we can finalize
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                # Attempt to extract code
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # If still more function calls remain
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        # optionally remove any trailing stop_words
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [run_execute(executor, code, args.prompt_type, data_name) for code in codes]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code_i = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result_i = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result_i]
        reports = [item[1] for item in result_i]

        # fix multiple-choice predictions if needed
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in ["A","B","C","D","E"]:
                # Possibly transform the output
                # e.g. a simple function choice_answer_clean() could do that
                pass
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join([c for c in preds[j] if c in ["A", "B", "C", "D", "E"]])

        sample.pop("prompt")
        sample.update({"code": code_i, "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minute"] = f"{int(time_use // 60)}:{int(time_use % 60):02d}"

    with open(out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)

    return result_json


def setup(args):
    # Initialize the client for your inference endpoint.
    client = setup_inference_client(args)

    # Evaluate on all data sets
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(client, data_name, args))

    # Add "avg" result
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # Print results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)