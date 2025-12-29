"""
프롬프트 템플릿 모듈

유형별 프롬프트 템플릿 정의:
- CoT (Chain of Thought) 프롬프트
- 기본 프롬프트 (숫자만 출력)
- 시스템 프롬프트
"""
from typing import Any

from api_inference.prompts.question_type import QuestionType, classify_question_type_with_llm
from common.prompts.system import SYSTEM_PROMPT as COMMON_SYSTEM_PROMPT


# =============================================================================
# CoT (Chain of Thought) 프롬프트
# 추론 과정을 먼저 서술하고, 마지막에 "정답: {숫자}" 형식으로 출력
# =============================================================================

COT_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

위 문제를 풀기 위해 다음 단계를 따르세요:
1. 지문의 핵심 내용을 파악하세요.
2. 각 선택지가 사실과 일치하는지 하나씩 분석하세요.
3. 분석 결과를 바탕으로 정답을 도출하세요.

추론 과정을 먼저 서술한 후, 반드시 마지막 줄에 "정답: 숫자" 형식으로 답을 출력하세요.
"""

COT_MULTI_LABEL_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 ㄱ, ㄴ, ㄷ, ㄹ 등의 보기 중 옳은 것을 모두 고르는 유형입니다.

다음 단계를 따라 풀이하세요:
1. 지문의 핵심 내용을 파악하세요.
2. 각 보기(ㄱ, ㄴ, ㄷ, ㄹ)의 내용을 지문에 비추어 옳은지 하나씩 판단하세요.
3. 옳은 보기들을 정리하세요.
4. 옳은 보기들의 조합에 해당하는 선택지 번호를 찾으세요. 선택지에 존재하지 않는다면 문제를 처음부터 다시 풀이하세요.

추론 과정을 먼저 서술한 후, 반드시 마지막 줄에 "정답: 숫자" 형식으로 답을 출력하세요.
"""

COT_SINGLE_CORRECT_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 선택지 중 옳은 것 또는 옳지 않은 것을 고르는 유형입니다.

다음 단계를 따라 풀이하세요:
1. 지문의 핵심 내용을 파악하세요.
2. 질문이 "옳은 것"을 찾는지 "옳지 않은 것"을 찾는지 확인하세요.
3. 각 선택지가 사실과 일치하는지 하나씩 분석하세요.
4. 질문의 요구에 맞는 선택지를 찾으세요.

추론 과정을 먼저 서술한 후, 반드시 마지막 줄에 "정답: 숫자" 형식으로 답을 출력하세요.
"""

COT_SEQUENCE_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 사건이나 내용을 시간 순서대로 나열하는 유형입니다.

다음 단계를 따라 풀이하세요:
1. 지문에서 각 사건(가), (나), (다), (라)의 내용을 파악하세요.
2. 역사적 사실과 지문을 바탕으로 각 사건의 시기를 추정하세요.
3. 시간 순서대로 정렬하여 올바른 배열을 찾으세요.

추론 과정을 먼저 서술한 후, 반드시 마지막 줄에 "정답: 숫자" 형식으로 답을 출력하세요.
"""

COT_FILL_BLANK_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 빈칸에 들어갈 알맞은 내용을 고르는 유형입니다.

다음 단계를 따라 풀이하세요:
1. 지문의 전체적인 맥락과 흐름을 파악하세요.
2. 빈칸 앞뒤의 문맥을 분석하세요.
3. 각 선택지를 빈칸에 대입하여 가장 적절한 것을 찾으세요.

추론 과정을 먼저 서술한 후, 반드시 마지막 줄에 "정답: 숫자" 형식으로 답을 출력하세요.
"""

COT_FACTUAL_RETRIEVAL_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 지문에서 구체적인 정보(무엇, 누구, 언제, 어디 등)를 확인하는 유형입니다.
지문에서 해당 정보를 찾아 정답을 도출하세요.
추론 과정을 먼저 서술한 후, 반드시 마지막 줄에 "정답: 숫자" 형식으로 답을 출력하세요.
"""

COT_REASONING_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 인과관계, 이유, 근거를 묻는 유형입니다.
지문에서 원인과 결과, 근거를 논리적으로 분석하여 정답을 도출하세요.
추론 과정을 먼저 서술한 후, 반드시 마지막 줄에 "정답: 숫자" 형식으로 답을 출력하세요.
"""

COT_CALCULATION_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 수치, 금액, 계산, 점유율 등을 묻는 계산형 문제입니다.
지문과 선택지의 수치를 비교·계산하여 정답을 도출하세요.
추론 과정을 먼저 서술한 후, 반드시 마지막 줄에 "정답: 숫자" 형식으로 답을 출력하세요.
"""

COT_SENTENCE_COMPLETION_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 문장을 완성하는 유형입니다.
지문과 문맥을 고려하여 가장 자연스럽게 문장이 완성되는 선택지를 고르세요.
추론 과정을 먼저 서술한 후, 반드시 마지막 줄에 "정답: 숫자" 형식으로 답을 출력하세요.
"""

COT_TOPIC_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 글의 주제나 제목을 묻는 유형입니다.
지문의 전체 내용을 파악하여 가장 적절한 주제/제목을 선택하세요.
추론 과정을 먼저 서술한 후, 반드시 마지막 줄에 "정답: 숫자" 형식으로 답을 출력하세요.
"""

# CoT 유형별 프롬프트 매핑
COT_PROMPT_TEMPLATES = {
    QuestionType.MULTI_LABEL: COT_MULTI_LABEL_PROMPT_FORMAT,
    QuestionType.SINGLE_CORRECT: COT_SINGLE_CORRECT_PROMPT_FORMAT,
    QuestionType.SEQUENCE: COT_SEQUENCE_PROMPT_FORMAT,
    QuestionType.FILL_BLANK: COT_FILL_BLANK_PROMPT_FORMAT,
    QuestionType.DEFAULT: COT_PROMPT_FORMAT,
    QuestionType.FACTUAL_RETRIEVAL: COT_FACTUAL_RETRIEVAL_PROMPT_FORMAT,
    QuestionType.REASONING: COT_REASONING_PROMPT_FORMAT,
    QuestionType.CALCULATION: COT_CALCULATION_PROMPT_FORMAT,
    QuestionType.SENTENCE_COMPLETION: COT_SENTENCE_COMPLETION_PROMPT_FORMAT,
    QuestionType.TOPIC: COT_TOPIC_PROMPT_FORMAT,
}


# =============================================================================
# 기본 프롬프트 (숫자만 출력)
# =============================================================================

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

FACTUAL_RETRIEVAL_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 지문에서 구체적인 정보(무엇, 누구, 언제, 어디 등)를 확인하는 유형입니다.
지문에서 해당 정보를 찾아 정답을 도출하세요.
정답 숫자 하나만 출력하세요.
"""

REASONING_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 인과관계, 이유, 근거를 묻는 유형입니다.
지문에서 원인과 결과, 근거를 논리적으로 분석하여 정답을 도출하세요.
정답 숫자 하나만 출력하세요.
"""

CALCULATION_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 수치, 금액, 계산, 점유율 등을 묻는 계산형 문제입니다.
지문과 선택지의 수치를 비교·계산하여 정답을 도출하세요.
정답 숫자 하나만 출력하세요.
"""

SENTENCE_COMPLETION_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 글의 주제나 제목을 묻는 유형입니다.
지문의 전체 내용을 파악하여 가장 적절한 주제/제목을 선택하세요.
정답 숫자 하나만 출력하세요.
"""

TOPIC_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 문장을 완성하는 유형입니다.
지문과 문맥을 고려하여 가장 자연스럽게 문장이 완성되는 선택지를 고르세요.
정답 숫자 하나만 출력하세요.
"""

# 유형별 프롬프트 매핑
PROMPT_TEMPLATES = {
    QuestionType.MULTI_LABEL: MULTI_LABEL_PROMPT_FORMAT,
    QuestionType.SINGLE_CORRECT: SINGLE_CORRECT_PROMPT_FORMAT,
    QuestionType.SEQUENCE: SEQUENCE_PROMPT_FORMAT,
    QuestionType.FILL_BLANK: FILL_BLANK_PROMPT_FORMAT,
    QuestionType.DEFAULT: BASE_PROMPT_FORMAT,
    QuestionType.FACTUAL_RETRIEVAL: FACTUAL_RETRIEVAL_PROMPT_FORMAT,
    QuestionType.REASONING: REASONING_PROMPT_FORMAT,
    QuestionType.CALCULATION: CALCULATION_PROMPT_FORMAT,
    QuestionType.SENTENCE_COMPLETION: SENTENCE_COMPLETION_PROMPT_FORMAT,
    QuestionType.TOPIC: TOPIC_PROMPT_FORMAT,
}


# =============================================================================
# 시스템 프롬프트
# =============================================================================

SYSTEM_PROMPTS = {
    QuestionType.MULTI_LABEL: "지문을 읽고 각 보기의 옳고 그름을 판단하여 질문에 답하세요.",
    QuestionType.SINGLE_CORRECT: "지문을 읽고 옳은 것 또는 옳지 않은 것을 정확히 고르세요.",
    QuestionType.SEQUENCE: "지문을 읽고 시간 순서를 정확히 파악하여 답하세요.",
    QuestionType.FILL_BLANK: "지문을 읽고 빈칸에 들어갈 알맞은 내용을 고르세요.",
    QuestionType.FACTUAL_RETRIEVAL: "지문에서 구체적인 정보를 찾아 정확히 답하세요.",
    QuestionType.REASONING: "지문에서 원인, 근거, 이유를 논리적으로 분석하여 답하세요.",
    QuestionType.CALCULATION: "지문과 선택지의 수치를 비교·계산하여 답하세요.",
    QuestionType.SENTENCE_COMPLETION: "문맥상 가장 자연스럽게 문장이 완성되는 선택지를 고르세요.",
    QuestionType.TOPIC: "글의 주제나 제목을 파악하여 가장 적절한 선택지를 고르세요.",
    QuestionType.DEFAULT: COMMON_SYSTEM_PROMPT,
}


# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_prompt_template(question_type: QuestionType, use_cot: bool = False) -> str:
    """
    문제 유형과 CoT 사용 여부에 따른 프롬프트 템플릿 반환
    
    Args:
        question_type: 문제 유형
        use_cot: CoT 프롬프트 사용 여부
    
    Returns:
        프롬프트 템플릿 문자열
    """
    if use_cot:
        return COT_PROMPT_TEMPLATES.get(question_type, COT_PROMPT_FORMAT)
    return PROMPT_TEMPLATES.get(question_type, BASE_PROMPT_FORMAT)


def get_system_prompt(question_type: QuestionType = None) -> str:
    """
    문제 유형에 따른 시스템 프롬프트 반환
    
    Args:
        question_type: 문제 유형 (None이면 기본 프롬프트)
    
    Returns:
        시스템 프롬프트 문자열
    """
    if question_type is None:
        return COMMON_SYSTEM_PROMPT
    return SYSTEM_PROMPTS.get(question_type, COMMON_SYSTEM_PROMPT)


def format_question_message(
    paragraph: str,
    question: str,
    question_plus: Any,
    choices_list: list,
    question_type: QuestionType = None,
    use_cot: bool = False
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
        use_cot: True면 CoT 프롬프트 사용, False면 기본 프롬프트 사용
    
    Returns:
        포맷팅된 프롬프트 문자열
    """
    # 문제 유형이 지정되지 않으면 자동 분류
    if question_type is None:
        question_type = classify_question_type_with_llm(question, question_plus, choices_list)
    
    # 선택지 문자열 생성
    choices_str = "\n".join([f"{i + 1} - {choice}" for i, choice in enumerate(choices_list)])
    
    # <보기> 유무에 따른 질문 내용 구성
    if question_plus and str(question_plus).strip() and str(question_plus) != 'nan':
        question_content = f"질문:\n{question}\n\n<보기>\n{question_plus}"
    else:
        question_content = f"질문:\n{question}"
    
    # 프롬프트 템플릿 선택 (CoT 또는 기본)
    prompt_template = get_prompt_template(question_type, use_cot)
    
    return prompt_template.format(
        paragraph=paragraph,
        question_content=question_content,
        choices=choices_str
    )


def create_messages(
    user_message: str,
    system_prompt: str = None,
    question_type: QuestionType = None
) -> list[dict[str, str]]:
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
