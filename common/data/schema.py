from dataclasses import dataclass


@dataclass(slots=True)
class QAExample:
    paragraph: str
    question: str
    choices: list[str]
    answer: str | None = None
    question_plus: str | None = None