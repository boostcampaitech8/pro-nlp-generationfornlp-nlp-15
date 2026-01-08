from __future__ import annotations

from ast import literal_eval

import pandas as pd

from .schema import QAExample


def load_qa_examples_from_csv(file_path: str) -> list[QAExample]:
    df = pd.read_csv(file_path)

    if isinstance(df["problems"].iloc[0], str):
        df["problems"] = df["problems"].apply(literal_eval)

    examples: list[QAExample] = []

    for _, row in df.iterrows():
        problem: dict = row["problems"]

        # answer: int/float/NaN 섞여 들어올 수 있으니 여기서 정리
        answer = problem.get("answer")
        if pd.isna(answer):
            answer = None
        else:
            answer = str(answer).strip()

        # question_plus도 NaN이면 None 처리(옵션이지만 안전)
        question_plus = problem.get("question_plus")
        if pd.isna(question_plus):
            question_plus = None
        else:
            question_plus = str(question_plus)

        examples.append(
            QAExample(
                id=str(row["id"]),
                paragraph=str(row["paragraph"]),
                question=str(problem["question"]),
                choices=[str(c) for c in problem["choices"]],
                answer=answer,
                question_plus=question_plus,
            )
        )

    return examples
