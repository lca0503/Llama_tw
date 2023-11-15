import os
import fire
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np


def main(
    model_name: str="",
    chat_model_name: str="",
    input_dir: str="./results",
    output_dir: str="./figures"
):
    layer_module_list = [
        "self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj", "self_attn_o_proj",
        "mlp_gate_proj", "mlp_up_proj", "mlp_down_proj",
        "input_layernorm", "post_attention_layernorm",
    ]

    bins = 1000

    plt.figure(figsize=(20, 10))

    if "70b" in model_name and "70b" in chat_model_name:
        num_layers = 80
    elif "13b" in model_name and "13b" in chat_model_name:
        num_layers = 40
    elif "7b" in model_name and "7b" in chat_model_name:
        num_layers = 32

    os.makedirs(output_dir, exist_ok = True)
    with torch.no_grad():
        for l in tqdm(range(num_layers)):
            for k in layer_module_list:
                llama2_w = torch.load(f"{input_dir}/{model_name}/layers_{l}/{k}.pt")
                llama2_chat_w = torch.load(f"{input_dir}/{chat_model_name}/layers_{l}/{k}.pt")
                llama2_w = llama2_w.cpu().detach().numpy().flatten()
                llama2_chat_w = llama2_chat_w.cpu().detach().numpy().flatten()

                plt.title(f"layer-{l} {k} histogram")
                plt.hist(llama2_w, bins=bins, color="blue", alpha=0.5, label=model_name)
                plt.hist(llama2_chat_w, bins=bins, color="red", alpha=0.5, label=chat_model_name)
                plt.legend()
                plt.savefig(f"{output_dir}/layer-{l}-{k}.png", bbox_inches='tight', pad_inches=0.2)
                plt.clf()
    
    with torch.no_grad():
        for m in ["embed_tokens", "norm", "lm_head"]:
            llama2_w = torch.load(f"{input_dir}/{model_name}/{m}.pt")
            llama2_chat_w = torch.load(f"{input_dir}/{chat_model_name}/{m}.pt")
            llama2_w = llama2_w.cpu().detach().numpy().flatten()
            llama2_chat_w = llama2_chat_w.cpu().detach().numpy().flatten()
            plt.title(f"{m} histogram")
            plt.hist(llama2_w, bins=bins, color="blue", alpha=0.3, label=model_name)
            plt.hist(llama2_chat_w, bins=bins, color="red", alpha=0.3, label=chat_model_name)
            plt.legend()
            plt.savefig(f"{output_dir}/{m}.png", bbox_inches='tight', pad_inches=0.2)
            plt.clf()
    
    print("DONE")

    
if __name__ == "__main__":
    fire.Fire(main)
