import fire
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer


def main(
    model_name: str="meta-llama/Llama-2-7b-chat-hf",
    peft_model_path: str="./results/llama-2-7b-chat-zh1B-lora/last",
    merged_model_path: str="llama-2-7b-chat-zh1B-lora-merged",
):
    chat_model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    chat_model = PeftModel.from_pretrained(chat_model, peft_model_path, torch_dtype="auto")

    merged_model = chat_model.merge_and_unload()
    merged_model.save_pretrained(merged_model_path)

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(merged_model_path)


if __name__ == "__main__":
    fire.Fire(main)
