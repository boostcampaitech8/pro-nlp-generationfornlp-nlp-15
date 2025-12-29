"""
문제 유형 정의 및 분류 모듈

QuestionType: 문제 유형 열거형
classify_question_type: 규칙 기반 문제 유형 분류 함수
"""
import re
from enum import Enum
from typing import List, Any
import asyncio
from api_inference.utils import AsyncAPIClient

class QuestionType(Enum):
    """문제 유형 열거형"""
    FACTUAL_RETRIEVAL = "factual_retrieval" # 1 구체적 정보 확인형 987
    SINGLE_CORRECT = "single_correct"     # 2 옳은 것/옳지 않은 것 단일 선택 431
    REASONING = "reasoning"               # 3 인과/근거형 224
    CALCULATION = "calculation"           # 4 수치/계산형 220	
    SENTENCE_COMPLETION = "sentence_completion" # 5 문장 완성형 94	
    FILL_BLANK = "fill_blank"             # 6 빈칸 채우기 33
    TOPIC = "topic"                       # 7 주제/제목형 26
    SEQUENCE = "sequence"                 # 8 시간 순서대로 나열하기 12
    MULTI_LABEL = "multi_label"           # 9 옳은 것 모두 고르기 (ㄱ, ㄴ, ㄷ 형) 4
    DEFAULT = "default"                   # 10 기본 유형


async def classify_question_type_with_llm(
    client: AsyncAPIClient,
    question: str,
    question_plus: Any,
    choices: List[str],
    system_prompt: str,
    semaphore: asyncio.Semaphore
) -> QuestionType:
    """
    LLM을 사용하여 문제 유형을 분류합니다.
    유형 분류 기준:
    1	FACTUAL_RETRIEVAL	구체적 정보 확인형	'무엇', '누구', '언제', '어디' 등을 묻는 사실 확인 질문
    2	SINGLE_CORRECT	단일 정답형     '옳은/옳지 않은' 것을 고르는 전형적인 객관식 유형
    3	REASONING	인과/근거형	    이유, 원인, 근거를 묻는 질문
    4	CALCULATION	수치/계산형     수치, 금액, 계산, 점유율 등을 묻는 질문
    5	SENTENCE_COMPLETION	문장 완성형     질문이 '?'로 끝나지 않고 선택지로 문장을 완성하는 유형
    6	FILL_BLANK	빈칸 추론형	    '(가)에', '밑줄 친' 등이 포함된 질문
    7	TOPIC	주제/제목형 	글의 주제나 제목을 묻는 질문
    8	SEQUENCE	순서 배열형	    사건의 순서나 나열을 묻는 질문
    9	MULTI_LABEL	합답형	'ㄱ, ㄴ' 등 보기를 조합하여 고르는 유형
    10	DEFAULT	기본 유형	-	위 유형에 해당하지 않는 기본 객관식 질문

    Args:
        client: 비동기 API 클라이언트
        question: 질문 텍스트
        question_plus: 보기 텍스트
        choices: 선택지 리스트
        system_prompt: LLM에 전달할 시스템 프롬프트
        semaphore: 동시 요청 제한용 세마포어

    Returns:
        QuestionType 열거형 값
    """

    async with semaphore:
        # LLM에 보낼 메시지 구성
        user_message = {
            "role": "user",
            "content": (
                f"질문: {question}\n"
                f"보기: {question_plus}\n"
                f"선택지: {', '.join(choices)}\n"
                "위 질문의 유형을 다음 중 하나로 분류하세요: \n"
                "- factual_retrieval: 구체적 정보 확인형. '무엇', '누구', '언제', '어디' 등을 묻는 사실 확인 질문.\n"
                "- single_correct: 단일 정답형. '옳은/옳지 않은' 것을 고르는 전형적인 객관식 유형.\n"
                "- reasoning: 인과/근거형. 이유, 원인, 근거를 묻는 질문.\n"
                "- calculation: 수치/계산형. 수치, 금액, 계산, 점유율 등을 묻는 질문.\n"
                "- sentence_completion: 문장 완성형. 질문이 '?'로 끝나지 않고 선택지로 문장을 완성하는 유형.\n"
                "- fill_blank: 빈칸 추론형. '(가)에', '밑줄 친' 등과 같이 빈칸에 들어갈 내용을 추론하는 질문.\n"
                "- topic: 주제/제목형. 글의 주제나 제목을 묻는 질문.\n"
                "- sequence: 순서 배열형. 사건의 순서나 나열을 묻는 질문.\n"
                "- multi_label: 합답형. 'ㄱ, ㄴ' 등 보기를 조합하여 고르는 유형.\n"
                "- default: 기본 유형. 위 유형에 해당하지 않는 기본 객관식 질문.\n")
        }

        messages = [
            {"role": "system", "content": system_prompt},
            user_message
        ]

        # LLM 호출
        try:
            response = await client.chat_completion(messages=messages)
            response = response.strip().lower()

            # 응답을 기반으로 유형 매핑
            if "multi_label" in response:
                return QuestionType.MULTI_LABEL
            elif "single_correct" in response:
                return QuestionType.SINGLE_CORRECT
            elif "sequence" in response:
                return QuestionType.SEQUENCE
            elif "fill_blank" in response:
                return QuestionType.FILL_BLANK
            elif "reasoning" in response:
                return QuestionType.REASONING
            elif "calculation" in response:
                return QuestionType.CALCULATION
            elif "sentence_completion" in response:
                return QuestionType.SENTENCE_COMPLETION
            elif "factual_retrieval" in response:
                return QuestionType.FACTUAL_RETRIEVAL
            elif "topic" in response:
                return QuestionType.TOPIC
            else:
                return QuestionType.DEFAULT

        except Exception as e:
            # 오류 발생 시 기본 유형 반환
            return QuestionType.DEFAULT

# def classify_question_type(
#     question: str,
#     question_plus: Any,
#     choices: List[str]
# ) -> QuestionType:
#     """
#     문제 유형을 분류합니다.
    
#     유형 분류 기준:
#     1	FACTUAL_RETRIEVAL	구체적 정보 확인형	'무엇', '누구', '언제', '어디' 등을 묻는 사실 확인 질문
#     2	SINGLE_CORRECT	단일 정답형     '옳은/옳지 않은' 것을 고르는 전형적인 객관식 유형
#     3	REASONING	인과/근거형	    이유, 원인, 근거를 묻는 질문
#     4	CALCULATION	수치/계산형     수치, 금액, 계산, 점유율 등을 묻는 질문
#     5	SENTENCE_COMPLETION	문장 완성형     질문이 '?'로 끝나지 않고 선택지로 문장을 완성하는 유형
#     6	FILL_BLANK	빈칸 추론형	    '(가)에', '밑줄 친' 등이 포함된 질문
#     7	TOPIC	주제/제목형 	글의 주제나 제목을 묻는 질문
#     8	SEQUENCE	순서 배열형	    사건의 순서나 나열을 묻는 질문
#     9	MULTI_LABEL	합답형	'ㄱ, ㄴ' 등 보기를 조합하여 고르는 유형
#     10	DEFAULT	기본 유형	-	위 유형에 해당하지 않는 기본 객관식 질문
    
#     Args:
#         question: 질문 텍스트
#         question_plus: <보기> 텍스트
#         choices: 선택지 리스트
    
#     Returns:
#         QuestionType 열거형 값
#     """
#     question_lower = question.lower() if question else ""
#     question_plus_str = str(question_plus) if question_plus and str(question_plus) != 'nan' else ""
#     choices_str = " ".join(choices) if choices else ""
    
#     # ===== 1. MULTI_LABEL 유형 검사 =====
#     # ㄱ, ㄴ, ㄷ, ㄹ 형태의 보기가 있는지 확인
#     korean_markers = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ']
#     has_korean_markers_in_question_plus = any(
#         marker in question_plus_str for marker in korean_markers
#     )
    
#     # 선택지가 ㄱ, ㄴ, ㄷ의 조합인지 확인 (예: "ㄱ, ㄷ", "ㄴ, ㄹ")
#     combination_pattern = r'[ㄱㄴㄷㄹㅁ]\s*,\s*[ㄱㄴㄷㄹㅁ]'
#     has_combination_choices = any(
#         re.search(combination_pattern, choice) for choice in choices
#     )
    
#     if has_korean_markers_in_question_plus and has_combination_choices:
#         return QuestionType.MULTI_LABEL
    
#     # ===== 2. SEQUENCE 유형 검사 =====
#     # 순서 관련 키워드 확인
#     sequence_keywords = ['순서', '나열', '시간순', '일어난 순', '발생 순', '시대순', '연대순']
#     has_sequence_keyword = any(kw in question_lower for kw in sequence_keywords)
    
#     # 선택지가 (가)→(나)→(다) 형태인지 확인
#     arrow_pattern = r'\([가나다라마]\)\s*[→\-]\s*\([가나다라마]\)'
#     has_arrow_choices = any(
#         re.search(arrow_pattern, choice) for choice in choices
#     )
    
#     if has_sequence_keyword or has_arrow_choices:
#         return QuestionType.SEQUENCE
    
#     # ===== 3. FILL_BLANK 유형 검사 =====
#     # 빈칸 관련 키워드 확인
#     fill_blank_keywords = ['빈칸', '들어갈', '밑줄', '괄호']
#     fill_blank_patterns = [
#         r'\([가나다라]\)\s*에\s*들어갈',
#         r'\([가나다라]\)\s*에\s*해당',
#         r'___',
#         r'\(\s*\)',
#     ]
    
#     has_fill_blank_keyword = any(kw in question_lower for kw in fill_blank_keywords)
#     has_fill_blank_pattern = any(
#         re.search(pattern, question) for pattern in fill_blank_patterns
#     )
    
#     if has_fill_blank_keyword or has_fill_blank_pattern:
#         return QuestionType.FILL_BLANK
    
#     # ===== 4. SINGLE_CORRECT 유형 검사 =====
#     # 옳은 것/옳지 않은 것 관련 키워드 확인
#     correct_keywords = [
#         '옳은 것', '옳지 않은 것', '올바른 것', '올바르지 않은 것',
#         '적절한 것', '적절하지 않은 것', '타당한 것', '타당하지 않은 것',
#         '맞는 것', '틀린 것', '거짓인 것', '참인 것',
#         '해당하는 것', '해당하지 않는 것'
#     ]
    
#     has_correct_keyword = any(kw in question for kw in correct_keywords)
    
#     if has_correct_keyword:
#         return QuestionType.SINGLE_CORRECT
    
#     # ===== 5. DEFAULT 유형 =====
#     return QuestionType.DEFAULT