import json
import os
import pickle
from pathlib import Path

import fire
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams


def main(
    model_name: str="",
    data_path: str="",
    max_tokens: int=512,
    temperature: float=0.1,
    top_p: float=0.9,
    tensor_parallel_size: int=8,
    batch_size: int=256,
    seed: int=42,
    output_dir: str="./results",
):
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    # Prepare vLLM model
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

    # Prepare NeuLab-TedTalks dataset
    with open(data_path, "r") as f:
        prompts = f.readlines()

    # Prompt model and get results
    all_sequences = []
    for idx in range(0, len(prompts), batch_size):
        batch_prompts = []
        for p in prompts[idx:idx+batch_size]:            
            if "chat" in model_name:
                batch_prompts += [f"[INST] {p.strip()} [/INST]"]
            else:
                batch_prompts += [p.strip()]

        ret = llm.generate(batch_prompts, sampling_params)
        all_sequences += [x.prompt_token_ids + x.outputs[0].token_ids for x in ret]
    
    # Save results
    os.makedirs(f"{output_dir}/{model_name}/", exist_ok=True)

    prompt_lang = data_path.split(".")[-1]
    with open(f"{output_dir}/{model_name}/repetition-{prompt_lang}.pkl", "wb") as fp:
        pickle.dump(all_sequences, fp)


if __name__ == "__main__":
    fire.Fire(main)
