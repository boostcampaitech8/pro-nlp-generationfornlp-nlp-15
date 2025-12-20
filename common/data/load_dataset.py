import logging
import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .read_csv import load_qa_examples_from_csv
from .message_builder import build_chat_messages
from .chat_tokenizer import tokenize_chat_dataset, filter_by_max_length

logger = logging.getLogger(__name__)


def load_tokenized_qa_dataset(
    file_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 1024,
    filter_by_max_length: bool = False,
    require_answer: bool = True,
) -> Dataset:
    
    dataset = load_qa_dataset(file_path, require_answer=require_answer)
    
    logger.info("[SFTDataLoader] Tokenizing")
    tokenized = tokenize_chat_dataset(dataset, tokenizer)

    if filter_by_max_length:
        logger.info(f"[SFTDataLoader] Filtering samples > {max_length} tokens")
        tokenized = filter_by_max_length(tokenized, max_length)

    return tokenized


def load_qa_dataset(
    file_path: str,
    require_answer: bool = True,
):
    logger.info("[SFTDataLoader] Loading QA examples")
    examples = load_qa_examples_from_csv(file_path)

    if require_answer:
        examples = [ex for ex in examples if ex.answer is not None]

    logger.info("[SFTDataLoader] Building chat messages")
    formatted: list[dict[str, list[dict[str, str]]]] = [
        build_chat_messages(example) for example in examples
    ]
    
    dataset = Dataset.from_list(formatted)

    return dataset
    