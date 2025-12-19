"""
데이터 로딩 및 프롬프트 생성 모듈
- test.csv 로딩 및 problems 파싱
- 기존 baseline/model_utils.py의 프롬프트 형식 재사용
"""
import pandas as pd
from ast import literal_eval
from typing import List, Dict, Tuple, Any


# 프롬프트 양식 (baseline/model_utils.py와 동일)
BASE_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
마지막에 적은 답을 정답으로 선택을 하니 마침표 나 숫자이외의 단어를 쓰지말고 정답 숫자 하나면 출력하면됩니다.
"""


def format_question_message(
    paragraph: str,
    question: str,
    question_plus: Any,
    choices_list: List[str]
) -> str:
    """
    <보기> 유무를 자동으로 판단하여 최종 프롬프트를 생성합니다.
    baseline/model_utils.py의 format_question_message 함수와 동일한 로직
    
    Args:
        paragraph: 지문 텍스트
        question: 질문 텍스트
        question_plus: <보기> 텍스트 (없을 수 있음)
        choices_list: 선택지 리스트
    
    Returns:
        포맷팅된 프롬프트 문자열
    """
    # 선택지 문자열 생성
    choices_str = "\n".join([f"{i + 1} - {choice}" for i, choice in enumerate(choices_list)])
    
    # <보기> 유무에 따른 질문 내용 구성
    if question_plus and str(question_plus).strip() and str(question_plus) != 'nan':
        question_content = f"질문:\n{question}\n\n<보기>\n{question_plus}"
    else:
        question_content = f"질문:\n{question}"
    
    return BASE_PROMPT_FORMAT.format(
        paragraph=paragraph,
        question_content=question_content,
        choices=choices_str
    )


def load_test_data(test_path: str) -> List[Dict[str, Any]]:
    """
    테스트 데이터 로드 및 파싱
    
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
            'num_choices': 선택지 개수 (4 또는 5)
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
        
        test_data.append({
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'question_plus': question_plus,
            'choices': problems['choices'],
            'num_choices': len(problems['choices'])
        })
    
    return test_data


def create_messages(
    user_message: str,
    system_prompt: str = "지문을 읽고 질문의 답을 구하세요."
) -> List[Dict[str, str]]:
    """
    API 요청용 메시지 리스트 생성
    
    Args:
        user_message: 사용자 메시지 (프롬프트)
        system_prompt: 시스템 프롬프트
    
    Returns:
        OpenAI API 형식의 메시지 리스트
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

