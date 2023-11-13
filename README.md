# Llama_tw

This is the branch for weighted loss. 

## Prepare data

While we cannot release our data due to our policy, we do provide our data preprocessing script. You can follow these steps:

1. Execute the [data/split_data.py](data/split_data.py) script to obtain a subset from the original dataset.
Here's an example of splitting a 1B subset from the Traditional Chinese Dataset.
```
python3 split_data.py --dataset_path pretrain_cht_test.jsonl --output_path pretrain_cht_test_1B.jsonl --split_length 1000000000
```

2. Run the [data/preprocess_dataset.py](data/preprocess_dataset.py) script to convert the JSONL file into a HuggingFace dataset."
```
python3 preprocess_dataset.py --dataset_path pretrain_cht_test_1B.jsonl --output_path pretrain_cht_test_1B
```


## Continual Pretraining

For our continual pretraining, we make use of [DeepSpeed](https://github.com/microsoft/DeepSpeed) Integration with the HuggingFace [Trainer](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.Trainer). Here is the [tutorial](https://huggingface.co/docs/transformers/main_classes/deepspeed).
- Claimer: We do not employ [flash attention](https://github.com/Dao-AILab/flash-attention) in our continual pretraining due to the tensor core and instruction [issues](https://github.com/Dao-AILab/flash-attention/issues/148) with the V100.

Please begin by checking the [configs/pretrain/llama_2.py](configs/pretrain/llama_2.py) file.
After reviewing the configuration files, you can pretrain your own llama model by executing the [llama_pretrain.py](./llama_pretrain.py) script.

### Default

Here is an example of pretraining Llama-2-7b-chat on the 1B Traditional Chinese Dataset.
```
python -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES \
--node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT llama_pretrain.py \ 
--model_name meta-llama/Llama-2-7b-chat-hf --dataset_path ./data/pretrain_cht_test_1B \
--run_name llama-2-7b-chat-zh1B --output_dir ./results/llama-2-7b-chat-zh1B 
```

### Freeze layers

Here is an example of pretraining Llama-2-7b-chat on the 1B Traditional Chinese Dataset while freezing the first 10 layers of Llama-2-7b-chat.
```
python -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES \
--node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT llama_pretrain.py \
--model_name meta-llama/Llama-2-7b-chat-hf --dataset_path ./data/pretrain_cht_test_1B \
--freeze_layers 0 1 2 3 4 5 6 7 8 9 \
--run_name llama-2-7b-chat-zh1B-freeze-first-10 --output_dir ./results/llama-2-7b-chat-zh1B-freeze-first-10
```

### Freeze modules

To freeze the weights of specific modules, you should include additional code in the [llama_pretrain.py](./llama_pretrain.py) script.
Here is an example of pretraining Llama-2-7b-chat on the 1B Traditional Chinese Dataset while freezing mlp modules

First, add the following code to the [llama_pretrain.py](./llama_pretrain.py) script.
```
for idx in range(len(model.model.layers)):
    for param in model.model.layers[idx].mlp.parameters():
    	param.requires_grad = False
```

After making these changes, proceed to execute the [llama_pretrain.py](./llama_pretrain.py) script.
```
python -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES \
--node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT llama_pretrain.py \
--model_name meta-llama/Llama-2-7b-chat-hf --dataset_path ./data/pretrain_cht_test_1B \
--run_name llama-2-7b-chat-zh1B-freeze-mlp --output_dir ./results/llama-2-7b-chat-zh1B-freeze-mlp 
```

### Adapter

We use [PEFT](https://github.com/huggingface/peft) to implement continual pretraining for Llama with adapters. You can choose to pretrain your Llama model with `LORA`, and `IA3`. For further information, please check the [configs/pretrain/peft.py](configs/pretrain/peft.py) file. During continual pretraining with an adapter, only the adapter weights will be saved. The model weights will not be saved.
Here is an example of pretraining Llama-2-7b-chat with LORA adapter on the 1B Traditional Chinese Dataset.
```
python -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES \
--node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT llama_pretrain.py \
--model_name meta-llama/Llama-2-7b-chat-hf --dataset_path ./data/pretrain_cht_test_1B \
--use_peft=True --peft_method LORA \
--run_name llama-2-7b-chat-zh1B-lora --output_dir ./results/llama-2-7b-chat-zh1B-lora
```

### Weighted loss

We use [教育部4808個常用字](https://language.moe.gov.tw/001/Upload/Files/site_content/download/mandr/%E6%95%99%E8%82%B2%E9%83%A84808%E5%80%8B%E5%B8%B8%E7%94%A8%E5%AD%97.xls) (Ministry of Education's 4808 Commonly Used Traditional Chinese Characters) for preparing our loss weights. 
We design our loss weights by assigning additional weights to tokens representing traditional Chinese characters, numerical values, and commonly used punctuation marks.
Here is an example to retrieve all tokens we need. 
```
python3 get_tokens.py
```
You should add these tokens to [utils/other_utils.py](utils/other_utils.py).

Here is an example of pretraining Llama-2-7b-chat with weighted loss on the 1B Traditional Chinese Dataset.
```
python -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES \
--node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT llama_pretrain_weighted_loss.py \
--model_name meta-llama/Llama-2-7b-chat-hf --dataset_path ./data/pretrain_cht_test_1B \
--use_peft=True --peft_method LORA \
--run_name llama-2-7b-chat-zh1B-lora --output_dir ./results/llama-2-7b-chat-zh1B-lora
```

### Caution

When conducting continual pretraining, the `add_eos_token` parameter in our tokenizer will be set to `True`. If you are performing inference on a model checkpoint in the middle of the training process, please ensure that you check your `tokenizer_config.json` file and set `add_eos_token` to `False` during inference.


## Inference

We also provide the [inference.py](inference.py) script for executing inferences with our model. We utilize [vLLM](https://github.com/vllm-project/vllm) to improve inference speed. You can customize the prompts inside the script according to your requirements.
Here is an example of using the [inference.py](inference.py) script.
```
python3 inference.py --model_name llama-2-7b-chat-zh1B \
--max_tokens 512 --temperature 0.1 --top_p 0.9 --tensor_parallel_size 8 --seed 42
```

When working with the [PEFT](https://github.com/huggingface/peft) model and conducting inference using [vLLM](https://github.com/vllm-project/vllm), it is necessary to first merge the model with the adapter weights. Here is an example of merging LORA weights with Llama-2-7b-chat-hf.
```
python3 merge_model.py --model_name meta-llama/Llama-2-7b-chat-hf \
--peft_model_path ./results/llama-2-7b-chat-zh1B-lora/last \
--merged_model_path llama-2-7b-chat-zh1B-lora-merged
```


## Analysis

For detailed instructions, please refer to each README.md file in the [analysis](./analysis) folder.


## References

We list some repositories that we have referenced.

- ***[facebookresearch/llama](https://github.com/facebookresearch/llama)***
- ***[facebookresearch/llama-recipes](https://github.com/facebookresearch/llama-recipes)***
- [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
- [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [ntunlplab/traditional-chinese-alpaca](https://github.com/ntunlplab/traditional-chinese-alpaca)
- [MiuLab/Taiwan-LLaMa](https://github.com/MiuLab/Taiwan-LLaMa)
- [ymcui/Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)


## Contact

If you have any questions, please do not hesitate to contact us at b08902123@csie.ntu.edu.tw