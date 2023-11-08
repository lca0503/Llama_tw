# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# https://github.com/facebookresearch/llama-recipes


from dataclasses import asdict

from peft import (AdaptionPromptConfig, IA3Config, LoraConfig,
                  PrefixTuningConfig)

from configs.pretrain.llama_2 import PRETRAIN_CONFIG
from configs.pretrain.peft import (ADAPTION_PROMPT_CONFIG, IA3_CONFIG,
                                   LORA_CONFIG, PREFIX_TUNING_CONFIG)


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, PRETRAIN_CONFIG):
                print(f"Warning: unknown parameter {k}")
                        
                        
def generate_peft_config(train_config, kwargs):
    configs = (LORA_CONFIG, ADAPTION_PROMPT_CONFIG, PREFIX_TUNING_CONFIG, IA3_CONFIG)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig, IA3Config)
    names = tuple(c.__name__.split("_CONFIG")[0] for c in configs)

    assert train_config.peft_method in names, f"Peft config not found: {train_config.peft_method}, not in {names}"
    
    config = configs[names.index(train_config.peft_method)]()
    
    update_config(config, **kwargs)
    params = asdict(config)
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)
    
    return peft_config
