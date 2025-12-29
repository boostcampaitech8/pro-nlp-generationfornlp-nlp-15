"""
데이터 로딩 모듈
 - LLM 기반 문제 유형 분류 및 유형별 프롬프트 생성
 - common 모듈의 기본 요소를 import하여 확장

common 모듈에서 가져오는 것:
 - load_qa_examples_from_csv: CSV 로더 (QAExample 반환)

api_inference.prompts 모듈에서 가져오는 것:
 - QuestionType: 문제 유형 열거형
 - classify_question_type_with_llm: LLM 기반 문제 유형 분류
 - format_question_message: 프롬프트 포맷터
 - SYSTEM_PROMPTS, PROMPT_TEMPLATES 등
"""
from typing import Any
from tqdm import tqdm

# ===== common 모듈에서 기본 요소 import =====
from common.data.read_csv import load_qa_examples_from_csv

# ===== api_inference.prompts 모듈에서 import =====
from api_inference.prompts import classify_question_type_with_llm

import pandas as pd

async def load_test_data(
    test_path: str,
    llm_client,
    system_prompt,
    semaphore,
    sample_size: int = None,
) -> list[dict[str, Any]]:
    """
    테스트 데이터 로드 및 파싱 (LLM 기반 문제 유형 분류, 비동기)
    sample_size 적용, csv 저장 옵션 분리
    """
    qa_examples = load_qa_examples_from_csv(test_path)
    if sample_size is not None and sample_size > 0:
        qa_examples = qa_examples[:sample_size]
    test_data = []
    for example in tqdm(qa_examples, desc="LLM 분류 진행"):
        question_type = await classify_question_type_with_llm(
            llm_client,
            example.question,
            example.question_plus,
            example.choices,
            system_prompt,
            semaphore,
        )
        test_data.append({
            "id": example.id,
            "paragraph": example.paragraph,
            "question": example.question,
            "question_plus": example.question_plus,
            "choices": example.choices,
            "num_choices": len(example.choices),
            "question_type": question_type,
            "answer": example.answer,
        })

    # csv 저장 (디버깅용)
    import time
    time_suffix = int(time.time()) % (10 ** 7)
    df = pd.DataFrame(test_data)
    save_cols = [col for col in ['id', 'question_type', 'question', 'answer'] if col in df.columns]
    df[save_cols].to_csv("data/classified_question_types.csv_{time_suffix}", index=False)
    return test_data
