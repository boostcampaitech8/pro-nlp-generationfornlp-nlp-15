from datasets import Dataset
from transformers import PreTrainedTokenizerBase


def tokenize_chat_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> Dataset:
    """
    Apply chat template and tokenize messages.
    """
    # kwargs for apply_chat_template
    # Some older transformers versions might error if unknown kwargs are passed, so we can try/except or just pass specific ones.
    # But Qwen 3 usually handles it.
    
    def tokenize_fn(examples: dict[str, list]) -> dict[str, list]:
        texts: list[str] = []
        for msg in examples["messages"]:
            # Try passing enable_thinking if supported (or if no error raised)
            try:
                text = tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking
                )
            except TypeError:
                # Fallback if tokenizer doesn't support 'enable_thinking' arg
                text = tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt
                )
            texts.append(text)

        outputs = tokenizer(
            texts,
            truncation=False,
            padding=False,
        )

        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )


def filter_by_max_length(dataset: Dataset, max_length: int) -> Dataset:
    """
    Filter samples exceeding max token length.
    """
    return dataset.filter(lambda x: len(x["input_ids"]) <= max_length)