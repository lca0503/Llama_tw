# Toxicity Analysis

In toxicity analysis, we utilize [ToxiGen](https://aclanthology.org/2022.acl-long.234/) as our benchmark dataset. 
(In line with [Llama-2's](https://arxiv.org/abs/2307.09288) methodology, we utilize a revised version of the dataset from [SafeNLP](https://aclanthology.org/2023.trustnlp-1.11))


## Prepare dataset

Our toxicity prompts are available in the [data](data/). You can directly use the provided prompts. English toxicity prompts are available in the [data/ToxiGen.json](data/ToxiGen.json). Traditional Chinese toxicity prompts are available in the [data/ToxiGen_tw.json](data/ToxiGen_tw.json)
Here is an overview of the process for preparing the toxicity dataset.
We download the dataset from [microsoft/SafeNLP](https://github.com/microsoft/SafeNLP).
Next, we prepare our Traditional Chinese toxicity dataset by executing [data/translate_toxigen.py](data/translate_toxigen.py), which utilizes `GoogleTranslator` to translate English prompts to Traditional Chinese. We implement it using the [deep-translator](https://pypi.org/project/deep-translator) package.
```
python translate_toxigen.py --data_path $DATA_PATH --output_path $OUTPUT_PATH
python translate_toxigen.py --data_path ToxiGen.json --output_path ToxiGen_tw.json
```


## Inference

Here are examples of performing inference on the toxicity benchmark dataset using [toxicity_vllm.py](toxicity_vllm.py).
```
python -u toxicity_vllm.py --model_name $MODEL_NAME --data_path $DATA_PATH --output_dir $OUTPUT_DIR 
python -u toxicity_vllm.py --model_name meta-llama/Llama-2-7b-chat-hf --data_path ./data/ToxiGen.json --output_dir ./results
python -u toxicity_vllm.py --model_name meta-llama/Llama-2-7b-chat-hf --data_path ./data/ToxiGen_tw.json --output_dir ./results
```
Your results will be saved at `${OUTPUT_DIR}/${MODEL_NAME}/toxigen.json` and `${OUTPUT_DIR}/${MODEL_NAME}/toxigen_tw.json` respectively.


## Evaluate 

Here is an example of evaluating the results.
```
python evaluate_toxicity.py --results_path $RESULTS_PATH
python evaluate_toxicity.py --results_path ./results/meta-llama/Llama-2-7b-chat-hf/toxigen.json
```
