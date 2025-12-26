"""
데이터 로딩 모듈
- 문제 유형 분류 및 유형별 프롬프트 생성
- common 모듈의 기본 요소를 import하여 확장

common 모듈에서 가져오는 것:
- load_qa_examples_from_csv: CSV 로더 (QAExample 반환)

api_inference.prompts 모듈에서 가져오는 것:
- QuestionType: 문제 유형 열거형
- classify_question_type: 규칙 기반 문제 유형 분류
- format_question_message: 프롬프트 포맷터
- SYSTEM_PROMPTS, PROMPT_TEMPLATES 등
"""
from typing import Any

# ===== common 모듈에서 기본 요소 import =====
from common.data.read_csv import load_qa_examples_from_csv

# ===== api_inference.prompts 모듈에서 import =====
from api_inference.prompts import classify_question_type


def load_test_data(test_path: str) -> list[dict[str, Any]]:
    """
    테스트 데이터 로드 및 파싱 (문제 유형 분류 포함)
    common/data/read_csv.py의 load_qa_examples_from_csv를 활용합니다.
    
    Args:
        test_path: test.csv 파일 경로

    Returns:
        파싱된 테스트 데이터 리스트
        각 항목: {
            'id': 문제 ID,
            'paragraph': 지문,
            'question': 질문,
            'question_plus': <보기> (있는 경우),
            'choices': 선택지 리스트,
            'num_choices': 선택지 개수 (4 또는 5),
            'question_type': 문제 유형 (QuestionType),
            'answer': 정답 (있는 경우)
        }
    """
    qa_examples = load_qa_examples_from_csv(test_path)

    test_data = []
    for idx, example in enumerate(qa_examples):
        question_type = classify_question_type(
            question=example.question,
            question_plus=example.question_plus,
            choices=example.choices
        )
        
        test_data.append({
            'id': example.id,  
            'paragraph': example.paragraph,
            'question': example.question,
            'question_plus': example.question_plus,
            'choices': example.choices,
            'num_choices': len(example.choices),
            'question_type': question_type,
            'answer': example.answer
        })
    
    return test_data
