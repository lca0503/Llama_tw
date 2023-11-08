import fire
import jsonlines
from datasets import Dataset


def main(
    dataset_path: str="./pretrain_cht_test_1B.jsonl", 
    output_path: str="./pretrain_cht_test_1B",
):

    with jsonlines.open(dataset_path, "r") as dataset_f:
        ds = Dataset.from_dict(
            {
                "text": [d["text"] for d in dataset_f]
            }
        )
    ds.save_to_disk(output_path)
        

if __name__ == "__main__":
    fire.Fire(main)
