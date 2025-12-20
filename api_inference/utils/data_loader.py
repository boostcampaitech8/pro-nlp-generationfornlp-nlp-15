"""
데이터 로딩 및 프롬프트 생성 모듈
- test.csv 로딩 및 problems 파싱
- 문제 유형 분류 및 유형별 프롬프트 생성
- 기존 baseline/model_utils.py의 프롬프트 형식 재사용
"""
import re
import pandas as pd
from ast import literal_eval
from typing import List, Dict, Tuple, Any
from enum import Enum


class QuestionType(Enum):
    """문제 유형 열거형"""
    MULTI_LABEL = "multi_label"           # 옳은 것 모두 고르기 (ㄱ, ㄴ, ㄷ 형)
    SINGLE_CORRECT = "single_correct"     # 옳은 것/옳지 않은 것 단일 선택
    SEQUENCE = "sequence"                  # 시간 순서대로 나열하기
    FILL_BLANK = "fill_blank"             # 빈칸 채우기
    DEFAULT = "default"                    # 기본 유형


# ===== 유형별 프롬프트 템플릿 =====

# 기본 프롬프트 (baseline/model_utils.py와 동일)
BASE_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
마지막에 적은 답을 정답으로 선택을 하니 마침표 나 숫자이외의 단어를 쓰지말고 정답 숫자 하나면 출력하면됩니다.
"""

# 옳은 것 모두 고르기 (multi-label) 프롬프트
MULTI_LABEL_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 ㄱ, ㄴ, ㄷ, ㄹ 등의 보기 중 옳은 것을 모두 고르는 유형입니다.
각 보기(ㄱ, ㄴ, ㄷ, ㄹ)의 내용이 지문에 비추어 옳은지 하나씩 판단한 후,
옳은 보기들의 조합에 해당하는 선택지 번호를 고르세요.
정답 숫자 하나만 출력하세요.
"""

# 옳은 것/옳지 않은 것 단일 선택 프롬프트
SINGLE_CORRECT_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 선택지 중 옳은 것 또는 옳지 않은 것을 고르는 유형입니다.
각 선택지가 지문의 내용과 일치하는지 확인하고, 질문에서 요구하는 것(옳은 것/옳지 않은 것)에 맞는 번호를 고르세요.
정답 숫자 하나만 출력하세요.
"""

# 시간 순서 나열 프롬프트
SEQUENCE_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 사건이나 내용을 시간 순서대로 나열하는 유형입니다.
지문과 역사적 사실을 바탕으로 (가), (나), (다), (라) 등의 순서를 파악하세요.
올바른 순서 배열에 해당하는 선택지 번호를 고르세요.
정답 숫자 하나만 출력하세요.
"""

# 빈칸 채우기 프롬프트
FILL_BLANK_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 빈칸에 들어갈 알맞은 내용을 고르는 유형입니다.
지문의 맥락과 문맥을 파악하여 빈칸 (가), (나) 등에 가장 적절한 내용을 선택하세요.
정답 숫자 하나만 출력하세요.
"""

# 유형별 프롬프트 매핑
PROMPT_TEMPLATES = {
    QuestionType.MULTI_LABEL: MULTI_LABEL_PROMPT_FORMAT,
    QuestionType.SINGLE_CORRECT: SINGLE_CORRECT_PROMPT_FORMAT,
    QuestionType.SEQUENCE: SEQUENCE_PROMPT_FORMAT,
    QuestionType.FILL_BLANK: FILL_BLANK_PROMPT_FORMAT,
    QuestionType.DEFAULT: BASE_PROMPT_FORMAT,
}

# 유형별 시스템 프롬프트
SYSTEM_PROMPTS = {
    QuestionType.MULTI_LABEL: "지문을 읽고 각 보기의 옳고 그름을 판단하여 질문에 답하세요.",
    QuestionType.SINGLE_CORRECT: "지문을 읽고 옳은 것 또는 옳지 않은 것을 정확히 고르세요.",
    QuestionType.SEQUENCE: "지문을 읽고 시간 순서를 정확히 파악하여 답하세요.",
    QuestionType.FILL_BLANK: "지문을 읽고 빈칸에 들어갈 알맞은 내용을 고르세요.",
    QuestionType.DEFAULT: "지문을 읽고 질문의 답을 구하세요.",
}


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


def format_question_message(
    paragraph: str,
    question: str,
    question_plus: Any,
    choices_list: List[str],
    question_type: QuestionType = None
) -> str:
    """
    <보기> 유무를 자동으로 판단하여 최종 프롬프트를 생성합니다.
    문제 유형에 따라 다른 프롬프트 템플릿을 사용합니다.
    
    Args:
        paragraph: 지문 텍스트
        question: 질문 텍스트
        question_plus: <보기> 텍스트 (없을 수 있음)
        choices_list: 선택지 리스트
        question_type: 문제 유형 (None이면 자동 분류)
    
    Returns:
        포맷팅된 프롬프트 문자열
    """
    # 문제 유형이 지정되지 않으면 자동 분류
    if question_type is None:
        question_type = classify_question_type(question, question_plus, choices_list)
    
    # 선택지 문자열 생성
    choices_str = "\n".join([f"{i + 1} - {choice}" for i, choice in enumerate(choices_list)])
    
    # <보기> 유무에 따른 질문 내용 구성
    if question_plus and str(question_plus).strip() and str(question_plus) != 'nan':
        question_content = f"질문:\n{question}\n\n<보기>\n{question_plus}"
    else:
        question_content = f"질문:\n{question}"
    
    # 유형별 프롬프트 템플릿 선택
    prompt_template = PROMPT_TEMPLATES.get(question_type, BASE_PROMPT_FORMAT)
    
    return prompt_template.format(
        paragraph=paragraph,
        question_content=question_content,
        choices=choices_str
    )


def load_test_data(test_path: str) -> List[Dict[str, Any]]:
    """
    테스트 데이터 로드 및 파싱 (문제 유형 분류 포함)
    
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
            'question_type': 문제 유형 (QuestionType)
        }
    """
    df = pd.read_csv(test_path)
    
    test_data = []
    for _, row in df.iterrows():
        problems = literal_eval(row['problems'])
        
        # question_plus 처리 (컬럼에 있거나 problems dict에 있을 수 있음)
        question_plus = None
        if 'question_plus' in row and pd.notna(row['question_plus']):
            question_plus = row['question_plus']
        elif 'question_plus' in problems and problems['question_plus']:
            question_plus = problems['question_plus']
        
        # 문제 유형 분류
        question_type = classify_question_type(
            question=problems['question'],
            question_plus=question_plus,
            choices=problems['choices']
        )
        
        test_data.append({
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'question_plus': question_plus,
            'choices': problems['choices'],
            'num_choices': len(problems['choices']),
            'question_type': question_type
        })
    
    return test_data


def create_messages(
    user_message: str,
    system_prompt: str = "지문을 읽고 질문의 답을 구하세요.",
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
    # 문제 유형이 지정되었고 시스템 프롬프트가 기본값이면 유형별 프롬프트 사용
    if question_type is not None and system_prompt == "지문을 읽고 질문의 답을 구하세요.":
        system_prompt = SYSTEM_PROMPTS.get(question_type, system_prompt)
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]


def get_question_type_stats(test_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    테스트 데이터의 문제 유형별 통계를 반환합니다.
    
    Args:
        test_data: load_test_data로 로드한 테스트 데이터
    
    Returns:
        유형별 문제 개수 딕셔너리
    """
    stats = {qt.value: 0 for qt in QuestionType}
    for item in test_data:
        question_type = item.get('question_type', QuestionType.DEFAULT)
        stats[question_type.value] += 1
    return stats

