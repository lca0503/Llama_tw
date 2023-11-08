import fire
import jsonlines
from transformers import LlamaTokenizer


def main(
    dataset_path: str="./pretrain_cht_test.jsonl", 
    output_path: str="./pretrain_cht_test_1B.jsonl",
    model_name: str="meta-llama/Llama-2-7b-chat-hf",
    split_length: int=1000000000,
):

    total_length = 0
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    sub_data = []

    with jsonlines.open(dataset_path, "r") as reader:
        for d in reader:
            total_length += len(tokenizer(d["text"], add_special_tokens=False).input_ids)
            sub_data.append(d)
            if total_length > split_length:
                print(total_length)
                break

    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(sub_data)


if __name__ == "__main__":
    fire.Fire(main)
