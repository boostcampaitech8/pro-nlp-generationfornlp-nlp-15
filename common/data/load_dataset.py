import logging
import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .read_data import load_qa_examples_from_file
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
    TRL의 completion_only_loss를 사용하기 위해 prompt/completion 형식으로 데이터셋 반환.

    TRL 0.26.1에서 completion_only_loss가 작동하려면:
    - 'prompt' 컬럼: system + user 메시지 (학습에서 제외됨)
    - 'completion' 컬럼: assistant 응답 (학습 대상)

    Returns:
        Dataset with 'prompt' and 'completion' columns.
    """
    logger.info("[SFTDataLoader] Loading QA examples")
    examples = load_qa_examples_from_file(file_path)

    if require_answer:
        examples = [ex for ex in examples if ex.answer is not None]

    logger.info("[SFTDataLoader] Building prompt/completion pairs")
    data: list[dict[str, str]] = []
    for example in examples:
        messages_dict = build_chat_messages(example)
        messages = messages_dict["messages"]

        # prompt: system + user messages
        # add_generation_prompt=True ensures we stop right before the assistant's turn
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # completion: derive by generating full text and subtracting prompt
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )

        if full_text.startswith(prompt):
            completion = full_text[len(prompt) :]
        else:
            # Fallback for templates that might change formatting slightly (rare but safe to handle)
            # If strictly non-matching, we just take the assistant content + eos
            assistant_content = next(
                (m["content"] for m in messages if m["role"] == "assistant"), ""
            )
            completion = assistant_content + tokenizer.eos_token

        data.append({"prompt": prompt, "completion": completion})

    dataset = Dataset.from_list(data)
    logger.info(
        f"[SFTDataLoader] Loaded {len(dataset)} samples with prompt/completion format"
    )

    return dataset


def load_tokenized_qa_dataset(
    file_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    require_answer: bool = True,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
    use_cot: bool = False,
    exclude_answer: bool = False,
) -> Dataset:

    logger.info("[SFTDataLoader] Loading QA examples")
    examples = load_qa_examples_from_file(file_path)

    if require_answer:
        examples = [ex for ex in examples if ex.answer is not None]

    logger.info(f"[SFTDataLoader] Building chat messages (use_cot={use_cot})")

    formatted: list[dict[str, list[dict[str, str]]]] = []
    for example in examples:
        msg_dict = build_chat_messages(example, use_cot=use_cot)

        # If exclude_answer is True, strip the assistant message containing the answer
        if exclude_answer:
            msg_dict["messages"] = [
                m for m in msg_dict["messages"] if m["role"] != "assistant"
            ]

        formatted.append(msg_dict)

    dataset = Dataset.from_list(formatted)

    logger.info(
        f"[SFTDataLoader] Tokenizing (add_generation_prompt={add_generation_prompt}, enable_thinking={enable_thinking})"
    )
    tokenized_ds = tokenize_chat_dataset(
        dataset,
        tokenizer,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )

    # Optional: Filter by max_length if needed. Infer.py passes max_length.
    # We should probably filter or at least warn?
    # For inference, truncation usually handled by tokenizer(truncation=False) in chat_tokenizer?
    # chat_tokenizer.py says truncation=False.
    # Let's filter to be safe if it's way too long, or usually just return.
    # But `filter_by_max_length` logic exists.
    # In my previous view, `load_tokenized_qa_dataset` had `do_filter_by_max_length` arg but it was removed?
    # The current signature has `max_length: int`.
    # I'll add filtering logic using max_length.

    # logger.info(f"[SFTDataLoader] Filtering samples > {max_length} tokens")
    # tokenized_ds = filter_by_max_length(tokenized_ds, max_length)

    return tokenized_ds
