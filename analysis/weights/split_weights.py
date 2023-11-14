import os
import fire
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

    
def main(
    model_name: str="",
    output_dir: str="./results"
):    
    layer_module_map = {
        "self_attn_q_proj": "self_attn.q_proj",
        "self_attn_k_proj": "self_attn.k_proj",
        "self_attn_v_proj": "self_attn.v_proj",
        "self_attn_o_proj": "self_attn.o_proj",
        "mlp_gate_proj": "mlp.gate_proj",
        "mlp_up_proj": "mlp.up_proj",
        "mlp_down_proj": "mlp.down_proj",
        "input_layernorm": "input_layernorm",
        "post_attention_layernorm": "post_attention_layernorm",
    }

    llama2 = LlamaForCausalLM.from_pretrained(f"{model_name}")
    llama2_weight_dir = f"{output_dir}/{model_name}"
    
    # Split all layers modules
    for idx in tqdm(range(len(llama2.model.layers))):
        os.makedirs(f"{llama2_weight_dir}/layers_{idx}", exist_ok=True)
        for k, v in layer_module_map.items():
            llama2_w = eval(f"llama2.model.layers[idx].{v}.weight")
            torch.save(llama2_w, f"{llama2_weight_dir}/layers_{idx}/{k}.pt")

    # Split other modules
    llama2_w = eval(f"llama2.model.embed_tokens.weight")
    torch.save(llama2_w, f"{llama2_weight_dir}/embed_tokens.pt")
    llama2_w = eval(f"llama2.model.norm.weight")
    torch.save(llama2_w, f"{llama2_weight_dir}/norm.pt")
    llama2_w = eval(f"llama2.lm_head.weight")
    torch.save(llama2_w, f"{llama2_weight_dir}/lm_head.pt")

if __name__ == "__main__":
    fire.Fire(main)
