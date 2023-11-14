import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import fire


def calculate_diff(x, y):
    #return torch.sum(torch.abs(x - y)).detach().cpu().numpy()
    #return torch.sum(torch.abs(x - y) / torch.norm(x)).detach().cpu().numpy()
    #epsilon = 1e-8
    #return torch.sum(torch.abs(y - x) / (torch.abs(x) + epsilon)).detach().cpu().numpy()
    return torch.nn.functional.cosine_similarity(x.flatten().unsqueeze(0), y.flatten().unsqueeze(0)).detach().cpu().numpy()
    

def plot_weights_diff(diff, output_path):
    x = list(range(len(diff)))

    plt.plot(x, diff, marker='o', linestyle='-')
    plt.xlabel("layers")
    plt.ylabel("diff")

    plt.savefig(output_path)
    plt.clf()
    

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

    if "70b" in model_name and "70b" in chat_model_name:
        num_layers = 80
    elif "13b" in model_name and "13b" in chat_model_name:
        num_layers = 40
    elif "7b" in model_name and "7b" in chat_model_name:
        num_layers = 32

    all_diffs = []
    with torch.no_grad():
        for l in tqdm(range(num_layers)):
            llama2_layer_w = []
            llama2_chat_layer_w = []
            for k in layer_module_list:
                llama2_w = torch.load(f"{input_dir}/{model_name}/layers_{l}/{k}.pt")
                llama2_chat_w = torch.load(f"{input_dir}/{chat_model_name}/layers_{l}/{k}.pt")
                llama2_w = llama2_w.cpu().flatten()
                llama2_chat_w = llama2_chat_w.cpu().flatten()
                llama2_layer_w.append(llama2_w)
                llama2_chat_layer_w.append(llama2_chat_w)
            diff = calculate_diff(torch.concatenate(llama2_layer_w), torch.concatenate(llama2_chat_layer_w))                
            all_diffs.append(diff)

    os.makedirs(output_dir, exist_ok = True)
    output_path = f"{output_dir}/layer_cosine_diff.png"
    plot_weights_diff(all_diffs, output_path)

    sorted_indices = [index for index, value in sorted(enumerate(all_diffs), key=lambda x: x[1])]
    print(sorted_indices[:10])
    
    print("DONE")

    
if __name__ == "__main__":
    fire.Fire(main)
