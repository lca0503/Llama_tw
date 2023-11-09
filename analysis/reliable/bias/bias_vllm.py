import json
import os
from pathlib import Path

import fire
import torch
from vllm import LLM, SamplingParams


def main(
    model_name: str="",
    data_dir: str="",
    max_tokens: int=512,
    temperature: float=0.1,
    top_p: float=0.9,
    tensor_parallel_size: int=8,
    batch_size: int=64,
    seed: int=42,
    output_dir: str="./results",
):

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    # Prepare vLLM model
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

    # Prepare BOLD dataset
    all_prompts_path = list(Path(data_dir).rglob("*.json"))
    all_prompts = {}
    for prompt_file in all_prompts_path:
        with open(prompt_file, 'r') as pf:
            raw_data = json.load(pf)
            sub_prompts = {category: [] for category in raw_data.keys()}
            for category, rd in raw_data.items():
                for name, prompts in rd.items():
                    sub_prompts[category] += prompts
            all_prompts.update(sub_prompts)

    # Prompt model and get results
    all_results = {category: [] for category in all_prompts.keys()}
    for category, prompts in all_prompts.items():
        for idx in range(0, len(prompts), batch_size):
            batch_prompts = []
            for p in prompts[idx:idx+batch_size]:            
                if "chat" in model_name:
                    batch_prompts += [f"[INST] {p.strip()} [/INST]"]
                else:
                    batch_prompts += [p.strip()]

            results = [x.outputs[0].text.strip() for x in llm.generate(batch_prompts, sampling_params)]
            assert len(results) == len(batch_prompts)
            
            for p, r in zip(prompts[idx:idx+batch_size], results):
                all_results[category].append(p.strip() + r)
        assert len(all_results[category]) == len(prompts)
        print("One category done!!!")

    # Save results
    os.makedirs(f"{output_dir}/{model_name}/", exist_ok=True)
    
    if "tw" in data_dir:
        output_file = "bold_tw.json"
    else:
        output_file = "bold.json"

    with open(f"{output_dir}/{model_name}/{output_file}", "w", encoding="utf-8") as outfile:
        json.dump(all_results, outfile, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
