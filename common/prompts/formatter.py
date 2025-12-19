# prompts/formatter.py

from .templates import BASE_PROMPT_FORMAT


def format_question_message(
    paragraph: str,
    question: str,
    question_plus: str | None,
    choices_list: list[str],
) -> str:
    """
    Build user prompt text for QA task.
    """
    choices_str: str = "\n".join(f"{i + 1} - {choice}" for i, choice in enumerate(choices_list))

    if question_plus and str(question_plus).strip() and str(question_plus) != "nan":
        question_content = f"질문:\n{question}\n\n<보기>\n{question_plus}"
    else:
        question_content = f"질문:\n{question}"

    return BASE_PROMPT_FORMAT.format(
        paragraph=paragraph,
        question_content=question_content,
        choices=choices_str,
    )
