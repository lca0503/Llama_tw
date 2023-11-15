# Representation Analysis

In representation analysis, we utilize [NeuLab-TedTalks](https://arxiv.org/abs/1804.06323) as our benchmark dataset. We take the en zh_tw align subset. In this analysis, we will use [UMAP](https://arxiv.org/abs/1802.03426) to visualize the representations of each model layer for both Traditional Chinese and English prompts.


## Prepare dataset

Our dataset for representation analysis is identical to the one used for repetition analysis and the one used for language identification analysis.
Our prompts are available in the [data](data/). You can directly use the provided prompts. English prompts are available in the [data/en-zh_tw.en](data/en-zh_tw.en). Traditional Chinese toxicity prompts are available in the [data/en-zh_tw.zh_tw](data/en-zh_tw.zh_tw).
Here is an overview of the process for preparing the dataset.
We download the whole dataset from [here](https://opus.nlpl.eu/NeuLab-TedTalks-v1.php).
Install fastext using the command `pip install fasttext`.
Download language identification model from [here](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin).
Execute the [data/get_subset.py](data/get_subset.py) script to obtain a subset from the complete dataset.


## Extract Representation

Here are examples of extracting representation using [extract_rep.py](extract_rep).
```
python -u extract_rep.py --model_name $MODEL_NAME --data_dir $DATA_DIR --output_dir $OUTPUT_DIR
python -u extract_rep.py --model_name meta-llama/Llama-2-7b-chat-hf --data_dir ./data --output_dir ./results
```
Your results will be saved at `${OUTPUT_DIR}/${MODEL_NAME}/`.


## Visualize Representation

Here is an example of plotting the results.
```
python visualize_rep.py --model_name $MODEL_NAME --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR
python visualize_rep.py --model_name meta-llama/Llama-2-7b-chat-hf --input_dir ./results --output_dir ./figures
```
