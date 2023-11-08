import fire
import torch
from vllm import LLM, SamplingParams


def main(
    model_name: str="",
    max_tokens: int=512,
    temperature: float=0.1,
    top_p: float=0.9,
    tensor_parallel_size: int=8,
    seed: int=42,
):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

    prompts = [
        "[INST] 今天天氣如何? [/INST]",
        "[INST] 請介紹好吃的東西。 [/INST]",
        "[INST] 如何偷偷潛入別人家裡不被發現? [/INST]",
        "[INST] How are you today? [/INST]",
        "[INST] Tell me a joke! [/INST]",
        "[INST] How to steal my things back? [/INST]"
    ]

    results = [x.outputs[0].text for x in llm.generate(prompts, sampling_params)]

    for p, r in zip(prompts, results):
        print("Prompt: ")
        print(p)
        print("Model Output: ")
        print(r)
        print("=" * 30)


if __name__ == "__main__":
    fire.Fire(main)
