# ARC_challenge

A commonsense reasoning benchmark. Please refer to this [paper](https://arxiv.org/abs/1803.05457).


## Inference and Evaluate

We perform this analysis using the [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) package.
To correctly install the package, please follow the guide provided on [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
Here is the example to run ARC_challenge benchmark.
For more information about running the [main.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/main.py) script, please check the [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
```
python main.py --model=hf-causal-experimental --model_args="pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True,max_length=4096" \
--tasks=arc_challenge --num_fewshot=25 --batch_size=2 --output_path="results/7b-chat/arc_challenge" --device cuda --no_cache
```