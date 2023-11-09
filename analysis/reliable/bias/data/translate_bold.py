import json
import os
from pathlib import Path

import fire
from deep_translator import GoogleTranslator
from tqdm import tqdm


def main(
    data_dir: str="./bold"
    output_dir: str="./bold_tw"
):
    translator = GoogleTranslator(source="auto", target="zh-TW")

    todo = list(Path(data_dir).rglob("*.json"))

    os.makedirs(output_dir, exist_ok=True)
    for p in tqdm(todo, "Processing all files: "):
        with open(p, "r") as infile:
            sub_d = json.load(infile)
            for k, v in tqdm(sub_d.items(), f"Processing {p.name}: "):
                for kk, vv in tqdm(v.items(), f"Processing {k}: "):
                    sub_d[k][kk] = translator.translate_batch(vv)
            with open(f"{output_dir}/{p.name}", "w", encoding="utf-8") as outfile:
                json.dump(sub_d, outfile, ensure_ascii=False)

                
if __name__ == '__main__':
    fire.Fire(main)
