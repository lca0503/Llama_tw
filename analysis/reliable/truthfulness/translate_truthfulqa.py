import fire
from datasets import load_dataset
from deep_translator import GoogleTranslator


def trans(example, translator):
    ret = example.copy()
    ret["question"] = translator.translate(example["question"])
    ret["mc1_targets"]["choices"] = translator.translate_batch(example["mc1_targets"]["choices"])
    ret['mc2_targets']["choices"] = translator.translate_batch(example["mc2_targets"]["choices"])

    return ret


def main(
    repo_name: str=""
):
    translator = GoogleTranslator(source="auto", target="zh-TW")

    truthfulqa = load_dataset("truthful_qa", "multiple_choice")

    truthfulqa_tw = truthfulqa.map(
        trans,
        fn_kwargs={"translator": translator}
    )
    
    truthfulqa_tw.push_to_hub(repo_name, private=True)


if __name__ == "__main__":
    fire.Fire(main)

