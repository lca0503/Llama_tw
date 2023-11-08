from datasets import Dataset


def tokenize_function(examples, tokenizer):
    return tokenizer([text for text in examples["text"]])


def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def get_pretrain_dataset(dataset_path, tokenizer, max_length, num_workers):
    pretrain_dataset = Dataset.load_from_disk(dataset_path)
    tokenized_pretrain_dataset = pretrain_dataset.map(
        tokenize_function,
        remove_columns=pretrain_dataset.column_names,
        batched=True,
        num_proc=num_workers,
        fn_kwargs={
            "tokenizer": tokenizer,
        }
    )
    lm_pretrain_dataset = tokenized_pretrain_dataset.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        fn_kwargs={
            "block_size": max_length
        }
    )

    return lm_pretrain_dataset
