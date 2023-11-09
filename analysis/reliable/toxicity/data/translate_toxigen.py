import json
import os

import fire
import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm


def main(
    data_path: str="./toxiGen.json"
    output_path: str="./toxiGen_tw.json"
):
    translator = GoogleTranslator(source="auto", target="zh-TW")

    df = pd.read_json(data_path)
    
    tqdm.pandas(desc="Translate: ")
    
    df["text"] = df["text"].progress_apply(translator.translate)

    result = df.to_json(orient="records")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json.loads(result), f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    fire.Fire(main)
