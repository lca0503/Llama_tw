from dataclasses import dataclass, field


@dataclass
class PRETRAIN_CONFIG:
    model_name: str="meta-llama/Llama-2-7b-chat-hf"
    dataset_path: str="./data/pretrain_cht_test_1B/"
    output_dir: str="./results/llama-2-7b-chat-zh1B"
    pretrain_ds_config: str="./configs/deepspeed/pretrain_ds_config.json"
    seed: int=42
    max_length: int=4096
    per_device_train_batch_size: int=1
    gradient_accumulation_steps: int=16 # 64 gpus
    num_train_epochs: int=1
    learning_rate: float=3e-5
    optim: str="adamw_torch"
    adam_beta1: float=0.9
    adam_beta2: float=0.95
    adam_epsilon: float=1e-5
    weight_decay: float=0.1
    max_grad_norm: float=1.0
    logging_steps: int=20
    save_steps: int=20
    save_total_limit: int=10
    report_to: str="tensorboard"
    run_name: str="llama-2-7b-chat-zh1B"
    fp16: bool=True
    gradient_checkpointing: bool=True
    use_cache: bool=False
    num_workers: int=32
    log_on_each_node: bool=False
    lr_scheduler_type: str="constant"
    resume_from_checkpoint: bool=False
    freeze_layers: list[int]=field(default_factory=list)
    peft_method: str=None # LORA, IA3
    use_peft: bool=False
