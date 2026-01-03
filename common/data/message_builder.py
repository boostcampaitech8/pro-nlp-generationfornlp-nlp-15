from ..prompts.formatter import format_question_message
from ..prompts.system import SYSTEM_PROMPT, SYSTEM_PROMPT_COT

from .schema import QAExample


def build_chat_messages(
    example: QAExample,
    use_cot: bool = False,
) -> dict[str, list[dict[str, str]]]:
    """
    Build chat-style messages from a QAExample.

    Handles two data formats:
    1. CSV format: paragraph, question, choices are all populated
    2. JSONL (CoT) format: paragraph is empty, question contains full formatted prompt

    CoT mode is automatically enabled if:
    - use_cot=True is explicitly set, OR
    - example.reasoning is not None (JSONL data with reasoning)
    """
    # Auto-detect CoT mode from reasoning presence
    is_cot_mode = use_cot or (example.reasoning is not None)

    # Check if this is JSONL format (paragraph empty, choices empty)
    # In this case, question already contains the full formatted prompt
    if not example.paragraph and not example.choices:
        user_message = example.question
    else:
        user_message = format_question_message(
            paragraph=example.paragraph,
            question=example.question,
            question_plus=example.question_plus,
            choices_list=example.choices,
        )

    system_content = SYSTEM_PROMPT_COT if is_cot_mode else SYSTEM_PROMPT

    # Gemma and some other models might not support 'system' role in chat template.
    # For CoT, we prepend the system instruction to the user message.
    if is_cot_mode:
        user_message = f"{system_content}\n\n{user_message}"
        messages: list[dict[str, str]] = [
            {"role": "user", "content": user_message},
        ]
    else:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ]

    if example.answer is not None:
        assistant_content = example.answer

        # CoT: Formatting Reasoning
        if example.reasoning is not None:
            # Gemma 3 or Generic CoT Style
            # Format: reasoning + answer
            assistant_content = f"{example.reasoning}\n\n정답: {example.answer}"

        messages.append({"role": "assistant", "content": assistant_content})

    return {"messages": messages}
