import logging
import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .read_csv import load_qa_examples_from_csv
from .message_builder import build_chat_messages
from .chat_tokenizer import tokenize_chat_dataset, filter_by_max_length

logger = logging.getLogger(__name__)


def load_sft_datasets(
    file_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 1024,
    split_ratio: float = 0.1,
    seed: int = 42,
    require_answer: bool = True,
) -> tuple[Dataset, Dataset]:
    """
    Prepare train/validation HuggingFace datasets for SFT.

    - require_answer=True  → training
    - require_answer=False → test / inference
    """
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

    logger.info(f"[SFTDataLoader] Filtering samples > {max_length} tokens")
    tokenized = filter_by_max_length(tokenized, max_length)

    split = tokenized.train_test_split(
        test_size=split_ratio,
        seed=seed,
    )

    return split["train"], split["test"]
