"""
api_inference 프롬프트 모듈

문제 유형 분류 및 유형별 프롬프트 템플릿 제공
"""
from api_inference.prompts.question_type import (
    QuestionType,
    classify_question_type,
)

from api_inference.prompts.templates import (
    # CoT 프롬프트
    COT_PROMPT_FORMAT,
    COT_MULTI_LABEL_PROMPT_FORMAT,
    COT_SINGLE_CORRECT_PROMPT_FORMAT,
    COT_SEQUENCE_PROMPT_FORMAT,
    COT_FILL_BLANK_PROMPT_FORMAT,
    COT_PROMPT_TEMPLATES,
    # 기본 프롬프트
    BASE_PROMPT_FORMAT,
    MULTI_LABEL_PROMPT_FORMAT,
    SINGLE_CORRECT_PROMPT_FORMAT,
    SEQUENCE_PROMPT_FORMAT,
    FILL_BLANK_PROMPT_FORMAT,
    PROMPT_TEMPLATES,
    # 시스템 프롬프트
    SYSTEM_PROMPTS,
    # 유틸리티 함수
    get_prompt_template,
    get_system_prompt,
    format_question_message,
)

__all__ = [
    # 문제 유형
    "QuestionType",
    "classify_question_type",
    # CoT 프롬프트
    "COT_PROMPT_FORMAT",
    "COT_MULTI_LABEL_PROMPT_FORMAT",
    "COT_SINGLE_CORRECT_PROMPT_FORMAT",
    "COT_SEQUENCE_PROMPT_FORMAT",
    "COT_FILL_BLANK_PROMPT_FORMAT",
    "COT_PROMPT_TEMPLATES",
    # 기본 프롬프트
    "BASE_PROMPT_FORMAT",
    "MULTI_LABEL_PROMPT_FORMAT",
    "SINGLE_CORRECT_PROMPT_FORMAT",
    "SEQUENCE_PROMPT_FORMAT",
    "FILL_BLANK_PROMPT_FORMAT",
    "PROMPT_TEMPLATES",
    # 시스템 프롬프트
    "SYSTEM_PROMPTS",
    # 유틸리티 함수
    "get_prompt_template",
    "get_system_prompt",
    "format_question_message",
]
