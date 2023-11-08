# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# https://github.com/facebookresearch/llama-recipes


from dataclasses import dataclass, field
from typing import List


@dataclass
class LORA_CONFIG:
     r: int=8
     lora_alpha: int=32
     target_modules: List[str]=field(default_factory=lambda: ["q_proj", "v_proj"])
     bias="none"
     task_type: str="CAUSAL_LM"
     lora_dropout: float=0.05
     inference_mode: bool=False


@dataclass
class IA3_CONFIG:
     target_modules: List[str]=field(default_factory=lambda: ["k_proj", "v_proj", "down_proj"])
     feedforward_modules: List[str]=field(default_factory=lambda: ["down_proj"])
     task_type: str="CAUSAL_LM"
