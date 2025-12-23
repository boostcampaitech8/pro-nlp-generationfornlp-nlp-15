import logging
import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .read_csv import load_qa_examples_from_csv
from .message_builder import build_chat_messages

logger = logging.getLogger(__name__)

    
def load_qa_dataset_tokenized(
    file_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = None,
    include_answer: bool = True,
):
    logger.info("[SFTDataLoader] Loading QA examples")
    examples = load_qa_examples_from_csv(file_path)

    if include_answer:
        for ex in examples:
            if ex.answer is None:
                raise ValueError("Found example without answer")

    logger.info("[SFTDataLoader] Building chat messages")
    message_dicts = [build_chat_messages(ex, include_answer=False) for ex in examples]
    
    dataset_items = []
    logger.info("[SFTDataLoader] Tokenizing Prompts")
    for msg_dict, ex in zip(message_dicts, examples):
        prompt = tokenizer.apply_chat_template(
            msg_dict["messages"],
            tokenize=False,
            add_generation_prompt=True
        )
        
        enc = tokenizer(
            prompt,
            truncation=False
        )
        
        data_dict = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
         
        if include_answer:
            data_dict.update({"answer": str(ex.answer)})

        dataset_items.append(data_dict)
        
    dataset = Dataset.from_list(dataset_items)
    
    if max_length:
        logger.info(f"[SFTDataLoader] Filtering samples > {max_length} tokens")
        dataset = filter_by_max_length(dataset, max_length)

    return dataset


def filter_by_max_length(dataset: Dataset, max_length: int) -> Dataset:
    """
    Filter samples exceeding max token length.
    """
    return dataset.filter(lambda x: len(x["input_ids"]) <= max_length)