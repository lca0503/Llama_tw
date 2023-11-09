# Repetition Analysis

In repetition analysis, we utilize [NeuLab-TedTalks](https://arxiv.org/abs/1804.06323) as our benchmark dataset. We take the en zh_tw align subset. 


## Prepare dataset

Our dataset for repetition analysis is identical to the one used for language identification analysis.
Our prompts are available in the [data](data/). You can directly use the provided prompts. English prompts are available in the [data/en-zh_tw.en](data/en-zh_tw.en). Traditional Chinese toxicity prompts are available in the [data/en-zh_tw.zh_tw](data/en-zh_tw.zh_tw).
Here is an overview of the process for preparing the dataset.
We download the whole dataset from [here](https://opus.nlpl.eu/NeuLab-TedTalks-v1.php).
Install fastext using the command `pip install fasttext`.
Download language identification model from [here](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin).
Execute the [data/get_subset.py](data/get_subset.py) script to obtain a subset from the complete dataset.


## Inference

Here are examples of performing inference on the repetition benchmark dataset using [repetition_vllm.py](repetition_vllm.py).
```
python -u repetition_vllm.py --model_name $MODEL_NAME --data_path $DATA_PATH  --output_dir $OUTPUT_DIR
python -u repetition_vllm.py --model_name meta-llama/Llama-2-7b-chat-hf --data_path ./data/en-zh_tw.en --output_dir ./results
python -u repetition_vllm.py --model_name meta-llama/Llama-2-7b-chat-hf --data_path ./data/en-zh_tw.zh_tw --output_dir ./results
```
Your results will be saved at `${OUTPUT_DIR}/${MODEL_NAME}/repetition-en.pkl` and `${OUTPUT_DIR}/${MODEL_NAME}/repetition-zh_tw.pkl` respectively.


## Evaluate

Here is an example of evaluating the results.
```
python evaluate_repetition.py --results_path $RESULTS_PATH
python evaluate_repetition.py --results_path ./results/meta-llama/Llama-2-7b-chat-hf/repetition-en.pkl
```
