# HellaSwag

A commonsense reasoning benchmark. Please refer to this [paper](https://arxiv.org/abs/1905.07830).


## Inference and Evaluate

We perform this analysis using the [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) package.
To correctly install the package, please follow the guide provided on [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
Here is the example to run HellaSwag benchmark.
For more information about running the [main.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/main.py) script, please check the [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
```
python main.py --model=hf-causal-experimental --model_args="pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True,max_length=4096" \
--tasks=hellaswag --num_fewshot=10 --batch_size=2 --output_path="results/7b-chat/hellaswag" --device cuda --no_cache
```