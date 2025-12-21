import logging
import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .read_csv import load_qa_examples_from_csv
from .message_builder import build_chat_messages
from .chat_tokenizer import tokenize_chat_dataset, filter_by_max_length

logger = logging.getLogger(__name__)


def load_text_qa_dataset(
    file_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    require_answer: bool = True,
) -> Dataset:
    """
    TRL의 response_template을 사용하기 위해 text 형식으로 데이터셋 반환.

    Returns:
        Dataset with 'text' column containing chat-formatted strings.
    """
    logger.info("[SFTDataLoader] Loading QA examples")
    examples = load_qa_examples_from_csv(file_path)

    if require_answer:
        examples = [ex for ex in examples if ex.answer is not None]

    logger.info("[SFTDataLoader] Building chat messages and applying template")
    texts: list[dict[str, str]] = []
    for example in examples:
        messages_dict = build_chat_messages(example)
        # chat template 적용하여 text로 변환
        text = tokenizer.apply_chat_template(
            messages_dict["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append({"text": text})

    dataset = Dataset.from_list(texts)
    logger.info(f"[SFTDataLoader] Loaded {len(dataset)} samples with text format")

    return dataset


def load_tokenized_qa_dataset(
    file_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 1024,
    do_filter_by_max_length: bool = False,
    require_answer: bool = True,
) -> Dataset:

    logger.info("[SFTDataLoader] Loading QA examples")
    examples = load_qa_examples_from_csv(file_path)

    if require_answer:
        examples = [ex for ex in examples if ex.answer is not None]

    logger.info("[SFTDataLoader] Building chat messages")
    formatted: list[dict[str, list[dict[str, str]]]] = [
        build_chat_messages(example) for example in examples
    ]

    dataset = Dataset.from_list(formatted)

    logger.info("[SFTDataLoader] Tokenizing")
    tokenized = tokenize_chat_dataset(dataset, tokenizer)

    if do_filter_by_max_length:
        logger.info(f"[SFTDataLoader] Filtering samples > {max_length} tokens")
        tokenized = filter_by_max_length(tokenized, max_length)

    return tokenized
