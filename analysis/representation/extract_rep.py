import os

import fire
import numpy
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(
    model_name: str="",
    data_dir: str="./data",
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    if "Llama" in model_name or "new" in model_name:
        if "7b" in model_name:
            layer_num = 32 + 1
        elif "13b" in model_name:
            layer_num = 40 + 1
        elif "70b" in model_name:
            layer_num = 80 + 1
        else:
            print("No implementation error!")
            exit()
    elif "bloom" in model_name:
        if "7b" in model_name:
            layer_num = 30 + 1
        else:
            print("No implementation error!")
            exit()
    else:
        print("No implementation error!")
        exit()

    os.makedirs(f"{output_dir}/{model_name}/en", exist_ok=True)
    os.makedirs(f"{output_dir}/{model_name}/tw", exist_ok=True)
    model.eval()
    
    en_representation = [[] for _ in range(layer_num)]
    with open(f"{data_dir}/en-zh_tw.en", "r") as f:
        en_prompts = f.readlines()
        for en_prompt in tqdm(en_prompts):
            with torch.no_grad():
                inputs = tokenizer(en_prompt, return_tensors="pt").to("cuda")
                hidden_states = model(**inputs, output_hidden_states=True).hidden_states
            for idx in range(layer_num):
                h = torch.squeeze(torch.mean(hidden_states[idx], dim=1))
                en_representation[idx].append(h.detach().cpu().numpy()
)
    for idx in range(layer_num):
        np.save(f"{output_dir}/{model_name}/en/layer-{idx}.npy", np.array(en_representation[idx]))
    
    zh_representation = [[] for _ in range(layer_num)]
    with open(f"{data_dir}/en-zh_tw.zh_tw", "r") as f:
        zh_prompts = f.readlines()
        for zh_prompt in tqdm(zh_prompts):
            with torch.no_grad():
                inputs = tokenizer(zh_prompt, return_tensors="pt").to("cuda")
                hidden_states = model(**inputs, output_hidden_states=True).hidden_states
            for idx in range(layer_num):
                h = torch.squeeze(torch.mean(hidden_states[idx], dim=1))
                zh_representation[idx].append(h.detach().cpu().numpy())
    for idx in range(layer_num):
        np.save(f"{output_dir}/{model_name}/tw/layer-{idx}.npy", np.array(zh_representation[idx]))


if __name__ == "__main__":
    fire.Fire(main)
