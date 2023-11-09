import pickle

import fasttext
import fire
from tqdm import tqdm
from transformers import LlamaTokenizer


def main(
    results_path: str="",
    lid_model_path: str="",
    model_name: str="meta-llama/Llama-2-7b-chat-hf",
):
    model = fasttext.load_model(lid_model_path)
    with open(results_path, "rb") as f:
        data = pickle.load(f)

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    zh_count = 0
    en_count = 0
    other_count = 0
    total_count = 0
    for whole_sequence in tqdm(data):
        text = tokenizer.decode(whole_sequence)
        pred = model.predict(text.replace("\n", " "))
        if pred[0][0] == "__label__zh":
            zh_count += 1
        elif pred[0][0] == "__label__en":
            en_count += 1
        else:
            other_count += 1
        total_count += 1
    print("=" * 30)
    print("en:", en_count / total_count, "zh_tw:", zh_count / total_count, "other:", other_count / total_count)


if __name__ == "__main__":
    fire.Fire(main)
