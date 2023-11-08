import fire
from peft import get_peft_model
from transformers import (DefaultDataCollator, LlamaForCausalLM,
                          LlamaTokenizer, Trainer, TrainingArguments)

from configs.pretrain.llama_2 import PRETRAIN_CONFIG
from utils.config_utils import generate_peft_config, update_config
from utils.peft_utils import SavePeftModelCallback
from utils.pretrain_dataset_utils import get_pretrain_dataset


def main(**kwargs):
    train_config = PRETRAIN_CONFIG()

    update_config((train_config), **kwargs)

    # Prepare Model
    model = LlamaForCausalLM.from_pretrained(train_config.model_name)
        
    model.config.use_cache = train_config.use_cache

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if train_config.use_peft:
        print("Use peft: ", train_config.peft_method)
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    if train_config.freeze_layers != []:
        print("Freeze layers: ", train_config.freeze_layers)
        for layer_idx in train_config.freeze_layers:
            for param in model.model.layers[layer_idx].parameters():
                param.requires_grad = False
                
    # Prepare Tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)

    tokenizer.add_eos_token = True

    # Prepare Dataset
    train_dataset = get_pretrain_dataset(
        dataset_path=train_config.dataset_path,
        tokenizer=tokenizer, 
        max_length=train_config.max_length,
        num_workers=train_config.num_workers
    )
    
    data_collator = DefaultDataCollator()

    # Prepare Trainer
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        seed=train_config.seed,
        deepspeed=train_config.pretrain_ds_config,
        optim=train_config.optim,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        num_train_epochs=train_config.num_train_epochs,
        learning_rate=train_config.learning_rate,
        lr_scheduler_type=train_config.lr_scheduler_type,
        adam_beta1=train_config.adam_beta1,
        adam_beta2=train_config.adam_beta2,
        adam_epsilon=train_config.adam_epsilon,
        weight_decay=train_config.weight_decay,
        max_grad_norm=train_config.max_grad_norm,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        report_to=train_config.report_to,
        run_name=train_config.run_name,
        fp16=train_config.fp16,
        gradient_checkpointing=train_config.gradient_checkpointing,
        log_on_each_node=train_config.log_on_each_node,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    if train_config.use_peft:
        trainer.add_callback(SavePeftModelCallback())
    
    trainer.train(resume_from_checkpoint=train_config.resume_from_checkpoint)
    
    trainer.save_model(f"{train_config.output_dir}/last")

    # Save original tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)    
    tokenizer.save_pretrained(f"{train_config.output_dir}/last")


if __name__ == "__main__":
    fire.Fire(main)
