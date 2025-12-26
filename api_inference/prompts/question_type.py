"""
문제 유형 정의 및 분류 모듈

QuestionType: 문제 유형 열거형
classify_question_type: 규칙 기반 문제 유형 분류 함수
"""
import re
from enum import Enum
from typing import List, Any


class QuestionType(Enum):
    """문제 유형 열거형 (api_inference 전용 확장)"""
    MULTI_LABEL = "multi_label"           # 옳은 것 모두 고르기 (ㄱ, ㄴ, ㄷ 형)
    SINGLE_CORRECT = "single_correct"     # 옳은 것/옳지 않은 것 단일 선택
    SEQUENCE = "sequence"                 # 시간 순서대로 나열하기
    FILL_BLANK = "fill_blank"             # 빈칸 채우기
    DEFAULT = "default"                   # 기본 유형


def classify_question_type(
    question: str,
    question_plus: Any,
    choices: List[str]
) -> QuestionType:
    """
    문제 유형을 분류합니다.
    
    유형 분류 기준:
    1. MULTI_LABEL: ㄱ, ㄴ, ㄷ 형 보기가 있고 선택지가 조합인 경우
    2. SINGLE_CORRECT: '옳은 것', '옳지 않은 것', '적절한 것' 등의 표현
    3. SEQUENCE: '순서', '나열', '시간순' 등의 표현이나 (가)→(나) 형태 선택지
    4. FILL_BLANK: '빈칸', '들어갈', '(가)에' 등의 표현
    5. DEFAULT: 위 유형에 해당하지 않는 경우
    
    Args:
        question: 질문 텍스트
        question_plus: <보기> 텍스트
        choices: 선택지 리스트
    
    Returns:
        QuestionType 열거형 값
    """
    question_lower = question.lower() if question else ""
    question_plus_str = str(question_plus) if question_plus and str(question_plus) != 'nan' else ""
    choices_str = " ".join(choices) if choices else ""
    
    # ===== 1. MULTI_LABEL 유형 검사 =====
    # ㄱ, ㄴ, ㄷ, ㄹ 형태의 보기가 있는지 확인
    korean_markers = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ']
    has_korean_markers_in_question_plus = any(
        marker in question_plus_str for marker in korean_markers
    )
    
    # 선택지가 ㄱ, ㄴ, ㄷ의 조합인지 확인 (예: "ㄱ, ㄷ", "ㄴ, ㄹ")
    combination_pattern = r'[ㄱㄴㄷㄹㅁ]\s*,\s*[ㄱㄴㄷㄹㅁ]'
    has_combination_choices = any(
        re.search(combination_pattern, choice) for choice in choices
    )
    
    if has_korean_markers_in_question_plus and has_combination_choices:
        return QuestionType.MULTI_LABEL
    
    # ===== 2. SEQUENCE 유형 검사 =====
    # 순서 관련 키워드 확인
    sequence_keywords = ['순서', '나열', '시간순', '일어난 순', '발생 순', '시대순', '연대순']
    has_sequence_keyword = any(kw in question_lower for kw in sequence_keywords)
    
    # 선택지가 (가)→(나)→(다) 형태인지 확인
    arrow_pattern = r'\([가나다라마]\)\s*[→\-]\s*\([가나다라마]\)'
    has_arrow_choices = any(
        re.search(arrow_pattern, choice) for choice in choices
    )
    
    if has_sequence_keyword or has_arrow_choices:
        return QuestionType.SEQUENCE
    
    # ===== 3. FILL_BLANK 유형 검사 =====
    # 빈칸 관련 키워드 확인
    fill_blank_keywords = ['빈칸', '들어갈', '밑줄', '괄호']
    fill_blank_patterns = [
        r'\([가나다라]\)\s*에\s*들어갈',
        r'\([가나다라]\)\s*에\s*해당',
        r'___',
        r'\(\s*\)',
    ]
    
    has_fill_blank_keyword = any(kw in question_lower for kw in fill_blank_keywords)
    has_fill_blank_pattern = any(
        re.search(pattern, question) for pattern in fill_blank_patterns
    )
    
    if has_fill_blank_keyword or has_fill_blank_pattern:
        return QuestionType.FILL_BLANK
    
    # ===== 4. SINGLE_CORRECT 유형 검사 =====
    # 옳은 것/옳지 않은 것 관련 키워드 확인
    correct_keywords = [
        '옳은 것', '옳지 않은 것', '올바른 것', '올바르지 않은 것',
        '적절한 것', '적절하지 않은 것', '타당한 것', '타당하지 않은 것',
        '맞는 것', '틀린 것', '거짓인 것', '참인 것',
        '해당하는 것', '해당하지 않는 것'
    ]
    
    has_correct_keyword = any(kw in question for kw in correct_keywords)
    
    if has_correct_keyword:
        return QuestionType.SINGLE_CORRECT
    
    # ===== 5. DEFAULT 유형 =====
    return QuestionType.DEFAULT
