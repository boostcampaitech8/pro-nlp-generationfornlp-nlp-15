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
from typing import List, Dict, Any

# ===== common 모듈에서 기본 요소 import =====
from common.prompts.system import SYSTEM_PROMPT as COMMON_SYSTEM_PROMPT
from common.data.read_csv import load_qa_examples_from_csv

# ===== api_inference.prompts 모듈에서 import =====
from api_inference.prompts import (
    QuestionType,
    classify_question_type,
    format_question_message,
    SYSTEM_PROMPTS,
    get_system_prompt,
)


def load_test_data(test_path: str) -> List[Dict[str, Any]]:
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


def create_messages(
    user_message: str,
    system_prompt: str = None,
    question_type: QuestionType = None
) -> List[Dict[str, str]]:
    """
    API 요청용 메시지 리스트 생성
    
    Args:
        user_message: 사용자 메시지 (프롬프트)
        system_prompt: 시스템 프롬프트 (None이면 유형별 기본값 사용)
        question_type: 문제 유형 (시스템 프롬프트 자동 선택용)
    
    Returns:
        OpenAI API 형식의 메시지 리스트
    """
    # 시스템 프롬프트 결정 로직
    if system_prompt is None:
        system_prompt = get_system_prompt(question_type)
    elif system_prompt == COMMON_SYSTEM_PROMPT and question_type is not None:
        # 기본 프롬프트가 전달되었지만 유형이 지정된 경우 유형별 프롬프트 사용
        system_prompt = SYSTEM_PROMPTS.get(question_type, system_prompt)
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]


