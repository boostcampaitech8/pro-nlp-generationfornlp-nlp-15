from __future__ import annotations

import json
from ast import literal_eval
import pandas as pd
from pathlib import Path

from .schema import QAExample


def load_qa_examples_from_file(file_path: str) -> list[QAExample]:
    path = Path(file_path)
    if path.suffix.lower() == ".jsonl":
        return _load_from_jsonl(file_path)
    else:
        return _load_from_csv(file_path)


def _load_from_csv(file_path: str) -> list[QAExample]:
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
                paragraph=str(row["paragraph"]),
                question=str(problem["question"]),
                choices=[str(c) for c in problem["choices"]],
                answer=answer,
                question_plus=question_plus,
            )
        )

    return examples


def _load_from_jsonl(file_path: str) -> list[QAExample]:
    examples: list[QAExample] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            # input: Pre-formatted string like "지문:\n...\n질문: ...\n선택지: ..."
            # We treat the entire input as the 'question' for simplicity in our message builder,
            # OR we try to parse it.
            # Given the message builder expects components, but we want to preserve the gold input format.
            # Strategy: Put everything in 'paragraph' or 'question' and leave others empty?
            # Better Strategy: Parse the input string to components if possible, OR
            # Adapt message_builder to handle "raw_prompt".

            # Let's try to parse the JSON output first
            output_str = data.get("output", "{}")
            try:
                output_data = json.loads(output_str)
            except json.JSONDecodeError:
                # Fallback if output is not JSON
                output_data = {}

            # Extract Reasoning
            # Structure: d.ek (dict), log.stance (str), log.critique (str), log.eval (list)
            reasoning_parts = []

            # 1. External Knowledge
            if "d" in output_data and "ek" in output_data["d"]:
                ek = output_data["d"]["ek"]
                if ek:
                    reasoning_parts.append("**관련 지식 (External Knowledge):**")
                    for k, v in ek.items():
                        reasoning_parts.append(f"- {k}: {v}")

            # 2. Logic / Stance
            if "log" in output_data:
                log = output_data["log"]

                if "stance" in log:
                    reasoning_parts.append(
                        f"\n**출제 의도 및 접근 (Stance):**\n{log['stance']}"
                    )

                if "critique" in log:
                    reasoning_parts.append(
                        f"\n**오답 분석 (Critique):**\n{log['critique']}"
                    )

                # 3. Evaluation Steps (Optional, slightly verbose, maybe skip or summarize)
                # Let's include if short, but 'critique' usually covers specific errors.

            reasoning_text = "\n".join(reasoning_parts) if reasoning_parts else None

            # Extract Answer
            # ans: {idx: int, txt: str}
            answer_idx = None
            if "ans" in output_data:
                # idx is 0-based or 1-based?
                # Sample: "idx": 2, "txt": "동녕부". Choices are 4.
                # Let's assume it matches the choices index.
                # In CSV loader, answer is usually 1-based index string '1', '2', '3', '4', '5'.
                # Let's check sample. Sample input choices: ['...', '...', '...', '...'] (list of strings)
                # Sample ans idx: 2.
                # If 0-indexed: 2 is 3rd option.
                # We need to standardize to what SFTTrainer/Collator expects.
                # CSV loader returns '1', '2', ...
                # So if idx is 0-based, we likely need to add 1.
                # Let's assume 0-based for now (standard for python lists in JSON).
                try:
                    answer_idx = str(
                        output_data["ans"]["idx"] + 1
                    )  # Convert to 1-based string
                except (KeyError, TypeError):
                    answer_idx = None

            # Input Parsing (Hack/Heuristic)
            # The input string is fully formatted: "지문:...\n질문: ...\n선택지:..."
            # Using this directly in 'paragraph' and leaving others None might work
            # if we adjust message_builder.

            inp = data.get("input", "")

            # For now, put full input in 'question' and others empty.
            # We will handle this in message_builder.

            examples.append(
                QAExample(
                    paragraph="",  # Empty to signal "use question as full prompt"
                    question=inp,  # Full formatted prompt
                    choices=[],  # Empty
                    answer=answer_idx,
                    question_plus=None,
                    reasoning=reasoning_text,
                    original_output=output_data,
                )
            )

    return examples
