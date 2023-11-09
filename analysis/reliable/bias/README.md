# Bias Analysis

In bias analysis, we utilize [BOLD](https://dl.acm.org/doi/10.1145/3442188.3445924) as our benchmark dataset. 


## Prepare dataset

Our bias prompts are available in the [data](data/). You can directly use the provided prompts. English bias prompts are available in the [data/bold](data/bold). Traditional Chinese bias prompts are available in the [data/bold_tw](data/bold_tw)
Here is an overview of the process for preparing the bias dataset.
We download the dataset from [amazon-science/bold](https://github.com/amazon-science/bold).
Next, we prepare our Traditional Chinese bias dataset by executing [data/translate_bold.py](data/translate_bold.py), which utilizes `GoogleTranslator` to translate English prompts to Traditional Chinese. We implement it using the [deep-translator](https://pypi.org/project/deep-translator) package.
```
python translate_bold.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR
python translate_bold.py --data_dir ./bold --output_dir ./bold_tw
```

## Inference

Here are examples of performing inference on the bias benchmark dataset using [bias_vllm.py](bias_vllm.py).
```
python -u bias_vllm.py --model_name $MODEL_NAME --data_dir $DATA_DIR  --output_dir $OUTPUT_DIR
python -u bias_vllm.py --model_name meta-llama/Llama-2-7b-chat-hf --data_dir ./data/bold/ --output_dir ./results
python -u bias_vllm.py --model_name meta-llama/Llama-2-7b-chat-hf --data_dir ./data/bold_tw/ --output_dir ./results
```
Your results will be saved at `${OUTPUT_DIR}/${MODEL_NAME}/bold.json` and `${OUTPUT_DIR}/${MODEL_NAME}/bold_tw.json` respectively.


## Evaluate 

Here is an example of evaluating the results.
```
python evaluate_bias.py --results_path $RESULTS_PATH 
python evaluate_bias.py --results_path ./results/meta-llama/Llama-2-7b-chat-hf/bold.json
```

(In line with [Llama-2's](https://arxiv.org/abs/2307.09288) methodology, we exclude prompts related to the religious ideology subgroups of Hinduism and Atheism.)
