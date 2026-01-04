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
from api_inference.prompts import classify_question_type_with_llm, QuestionType

import pandas as pd

async def load_test_data(
    test_path: str,
    llm_client,
    system_prompt,
    semaphore,
    sample_size: int = None,
    classification_votes: int = 1,
    skip_classification: bool = False,
) -> list[dict[str, Any]]:
    """
    테스트 데이터 로드 및 파싱 (LLM 기반 문제 유형 분류, 비동기)
    sample_size 적용, csv 저장 옵션 분리
    
    Args:
        test_path: 테스트 데이터 CSV 경로
        llm_client: 비동기 API 클라이언트
        system_prompt: 시스템 프롬프트
        semaphore: 동시 요청 제한용 세마포어
        sample_size: 샘플 수 제한 (None이면 전체)
        classification_votes: 문제 유형 분류 투표 횟수 (기본값: 1, 다수결 투표 시 3 이상 권장)
        skip_classification: True면 LLM 분류 스킵하고 DEFAULT 유형 사용
    """
    qa_examples = load_qa_examples_from_csv(test_path)
    if sample_size is not None and sample_size > 0:
        qa_examples = qa_examples[:sample_size]
    
    test_data = []
    
    if skip_classification:
        # LLM 분류 스킵 - 모든 문제를 DEFAULT 유형으로 설정
        for example in tqdm(qa_examples, desc="데이터 로딩 (분류 스킵)"):
            test_data.append({
                "id": example.id,
                "paragraph": example.paragraph,
                "question": example.question,
                "question_plus": example.question_plus,
                "choices": example.choices,
                "num_choices": len(example.choices),
                "question_type": QuestionType.DEFAULT,
                "answer": example.answer,
            })
    else:
        # LLM 기반 문제 유형 분류
        vote_desc = f"LLM 분류 진행 (votes={classification_votes})"
        for example in tqdm(qa_examples, desc=vote_desc):
            question_type = await classify_question_type_with_llm(
                llm_client,
                example.question,
                example.question_plus,
                example.choices,
                system_prompt,
                semaphore,
                num_votes=classification_votes,
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

        # csv 저장 (디버깅용) - 분류한 경우에만
        import time
        time_suffix = int(time.time()) % (10 ** 7)
        df = pd.DataFrame(test_data)
        # question_type을 문자열로 변환
        df['question_type'] = df['question_type'].apply(lambda x: x.value if hasattr(x, 'value') else str(x))
        save_cols = [col for col in ['id', 'question_type', 'question', 'answer'] if col in df.columns]
        df[save_cols].to_csv(f"data/classified_question_types_{time_suffix}.csv", index=False)
    
    return test_data
