from ..prompts.formatter import format_question_message
from ..prompts.system import SYSTEM_PROMPT, SYSTEM_PROMPT_COT

from .schema import QAExample


def build_chat_messages(
    example: QAExample,
    use_cot: bool = False,
) -> dict[str, list[dict[str, str]]]:
    """
    Build chat-style messages from a QAExample.
    """
    user_message: str = format_question_message(
        paragraph=example.paragraph,
        question=example.question,
        question_plus=example.question_plus,
        choices_list=example.choices,
    )

    system_content = SYSTEM_PROMPT_COT if use_cot else SYSTEM_PROMPT

    # Gemma and some other models might not support 'system' role in chat template.
    # To be safe, we prepend the system instruction to the user message.
    # Or strict adherence: check if we want to keep system role.
    # Given the user wants to ENFORCE it, let's prepend it to user message if use_cot is True.
    
    if use_cot:
        user_message = f"{system_content}\n\n{user_message}"
        # If we merged it, we only send user message
        messages: list[dict[str, str]] = [
            {"role": "user", "content": user_message},
        ]
    else:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ]

    if example.answer is not None:
        messages.append(
            {"role": "assistant", "content": example.answer}
        )

    return {"messages": messages}