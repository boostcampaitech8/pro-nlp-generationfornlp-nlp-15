from ..prompts.formatter import format_question_message
from ..prompts.system import SYSTEM_PROMPT

from .schema import QAExample


def build_chat_messages(example: QAExample) -> dict[str, list[dict[str, str]]]:
    """
    Build chat-style messages from a QAExample.
    """
    user_message: str = format_question_message(
        paragraph=example.paragraph,
        question=example.question,
        question_plus=example.question_plus,
        choices_list=example.choices,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if example.answer is not None:
        messages.append(
            {"role": "assistant", "content": example.answer}
        )

    return {"messages": messages}