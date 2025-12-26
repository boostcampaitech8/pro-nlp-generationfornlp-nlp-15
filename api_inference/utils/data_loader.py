"""
데이터 로딩 및 프롬프트 생성 모듈
- test.csv 로딩 및 problems 파싱
- 문제 유형 분류 및 유형별 프롬프트 생성
- common 모듈의 기본 요소를 import하여 확장

common 모듈에서 가져오는 것:
- BASE_PROMPT_FORMAT: 기본 프롬프트 템플릿
- SYSTEM_PROMPT: 기본 시스템 프롬프트  
- format_question_message: 기본 메시지 포맷터 (common_format_question_message로 alias)
- load_qa_examples_from_csv: CSV 로더 (QAExample 반환)
- QAExample: 데이터 클래스

api_inference에서 확장하는 것:
- QuestionType: 문제 유형 열거형
- 유형별 프롬프트 템플릿 (MULTI_LABEL, SINGLE_CORRECT, SEQUENCE, FILL_BLANK)
- classify_question_type: 규칙 기반 문제 유형 분류
"""
import re
from typing import List, Dict, Any
from enum import Enum

# ===== common 모듈에서 기본 요소 import =====
from common.prompts.templates import BASE_PROMPT_FORMAT as COMMON_BASE_PROMPT_FORMAT
from common.prompts.system import SYSTEM_PROMPT as COMMON_SYSTEM_PROMPT
from common.prompts.formatter import format_question_message as common_format_question_message
from common.data.read_csv import load_qa_examples_from_csv


class QuestionType(Enum):
    """문제 유형 열거형 (api_inference 전용 확장)"""
    MULTI_LABEL = "multi_label"           # 옳은 것 모두 고르기 (ㄱ, ㄴ, ㄷ 형)
    SINGLE_CORRECT = "single_correct"     # 옳은 것/옳지 않은 것 단일 선택
    SEQUENCE = "sequence"                  # 시간 순서대로 나열하기
    FILL_BLANK = "fill_blank"             # 빈칸 채우기
    DEFAULT = "default"                    # 기본 유형


# ===== 유형별 프롬프트 템플릿 (api_inference 전용 확장) =====

# ===== CoT (Chain of Thought) 프롬프트 =====
# 추론 과정을 먼저 서술하고, 마지막에 "정답: {숫자}" 형식으로 출력

COT_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

위 문제를 풀기 위해 다음 단계를 따르세요:
1. 지문의 핵심 내용을 파악하세요.
2. 각 선택지가 지문의 내용과 일치하는지 하나씩 분석하세요.
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
1. 각 보기(ㄱ, ㄴ, ㄷ, ㄹ)의 내용을 지문에 비추어 옳은지 하나씩 판단하세요.
2. 옳은 보기들을 정리하세요.
3. 옳은 보기들의 조합에 해당하는 선택지 번호를 찾으세요.

추론 과정을 먼저 서술한 후, 반드시 마지막 줄에 "정답: 숫자" 형식으로 답을 출력하세요.
"""

COT_SINGLE_CORRECT_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

이 문제는 선택지 중 옳은 것 또는 옳지 않은 것을 고르는 유형입니다.

다음 단계를 따라 풀이하세요:
1. 질문이 "옳은 것"을 찾는지 "옳지 않은 것"을 찾는지 확인하세요.
2. 각 선택지가 지문의 내용과 일치하는지 하나씩 분석하세요.
3. 질문의 요구에 맞는 선택지를 찾으세요.

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

# CoT 유형별 프롬프트 매핑
COT_PROMPT_TEMPLATES = {
    QuestionType.MULTI_LABEL: COT_MULTI_LABEL_PROMPT_FORMAT,
    QuestionType.SINGLE_CORRECT: COT_SINGLE_CORRECT_PROMPT_FORMAT,
    QuestionType.SEQUENCE: COT_SEQUENCE_PROMPT_FORMAT,
    QuestionType.FILL_BLANK: COT_FILL_BLANK_PROMPT_FORMAT,
    QuestionType.DEFAULT: COT_PROMPT_FORMAT,
}


# ===== 기본 프롬프트 (숫자만 출력) =====

# 기본 프롬프트 - common 기반으로 api_inference용 꼬리 문구 추가
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

# 유형별 시스템 프롬프트 (common의 SYSTEM_PROMPT 확장)
SYSTEM_PROMPTS = {
    QuestionType.MULTI_LABEL: "지문을 읽고 각 보기의 옳고 그름을 판단하여 질문에 답하세요.",
    QuestionType.SINGLE_CORRECT: "지문을 읽고 옳은 것 또는 옳지 않은 것을 정확히 고르세요.",
    QuestionType.SEQUENCE: "지문을 읽고 시간 순서를 정확히 파악하여 답하세요.",
    QuestionType.FILL_BLANK: "지문을 읽고 빈칸에 들어갈 알맞은 내용을 고르세요.",
    QuestionType.DEFAULT: COMMON_SYSTEM_PROMPT,  # common의 기본 시스템 프롬프트 사용
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
        question_type = classify_question_type(question, question_plus, choices_list)
    
    # 선택지 문자열 생성
    choices_str = "\n".join([f"{i + 1} - {choice}" for i, choice in enumerate(choices_list)])
    
    # <보기> 유무에 따른 질문 내용 구성
    if question_plus and str(question_plus).strip() and str(question_plus) != 'nan':
        question_content = f"질문:\n{question}\n\n<보기>\n{question_plus}"
    else:
        question_content = f"질문:\n{question}"
    
    # 프롬프트 템플릿 선택 (CoT 또는 기본)
    if use_cot:
        prompt_template = COT_PROMPT_TEMPLATES.get(question_type, COT_PROMPT_FORMAT)
    else:
        prompt_template = PROMPT_TEMPLATES.get(question_type, BASE_PROMPT_FORMAT)
    
    return prompt_template.format(
        paragraph=paragraph,
        question_content=question_content,
        choices=choices_str
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
            'id': example.id if example.id is not None else idx,  
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
        # 문제 유형이 지정되면 유형별 프롬프트, 아니면 common의 기본 프롬프트
        if question_type is not None:
            system_prompt = SYSTEM_PROMPTS.get(question_type, COMMON_SYSTEM_PROMPT)
        else:
            system_prompt = COMMON_SYSTEM_PROMPT
    elif system_prompt == COMMON_SYSTEM_PROMPT and question_type is not None:
        # 기본 프롬프트가 전달되었지만 유형이 지정된 경우 유형별 프롬프트 사용
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


def compute_f1_score(predictions: List[str], labels: List[str]) -> Dict[str, float]:
    """
    예측값과 정답 레이블로 Macro F1 Score 및 Accuracy를 계산합니다.
    
    Args:
        predictions: 예측 답변 리스트 ("1", "2", "3", "4", "5")
        labels: 정답 레이블 리스트 ("1", "2", "3", "4", "5")
    
    Returns:
        {
            'accuracy': 정확도,
            'f1_macro': Macro F1 Score,
            'f1_weighted': Weighted F1 Score,
            'correct': 맞은 개수,
            'total': 전체 개수
        }
    """
    from collections import Counter
    
    # 유효한 예측만 필터링 (answer가 "0"인 경우 파싱 실패)
    valid_pairs = [
        (pred, label) for pred, label in zip(predictions, labels)
        if pred != "0" and label is not None
    ]
    
    if not valid_pairs:
        return {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'correct': 0,
            'total': 0
        }
    
    valid_preds, valid_labels = zip(*valid_pairs)
    
    # Accuracy 계산
    correct = sum(1 for p, l in zip(valid_preds, valid_labels) if p == l)
    accuracy = correct / len(valid_pairs)
    
    # 모든 클래스 (1~5)
    all_classes = set(valid_preds) | set(valid_labels)
    
    # 클래스별 F1 계산
    f1_scores = []
    weighted_f1_scores = []
    label_counts = Counter(valid_labels)
    total_samples = len(valid_labels)
    
    for cls in all_classes:
        # True Positives, False Positives, False Negatives
        tp = sum(1 for p, l in zip(valid_preds, valid_labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(valid_preds, valid_labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(valid_preds, valid_labels) if p != cls and l == cls)
        
        # Precision, Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        
        # Weighted F1
        weight = label_counts[cls] / total_samples if cls in label_counts else 0
        weighted_f1_scores.append(f1 * weight)
    
    f1_macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    f1_weighted = sum(weighted_f1_scores)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'correct': correct,
        'total': len(valid_pairs)
    }


def compute_f1_by_question_type(
    results: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    문제 유형별 F1 Score를 계산합니다.
    
    Args:
        results: API 추론 결과 리스트 [{'id': ..., 'answer': ..., 'question_type': ...}, ...]
        test_data: 정답이 포함된 테스트 데이터 리스트
    
    Returns:
        유형별 metrics 딕셔너리
        {
            'multi_label': {'accuracy': ..., 'f1_macro': ..., ...},
            'single_correct': {...},
            ...
            'overall': {...}
        }
    """
    # ID로 정답 매핑
    id_to_answer = {item['id']: item.get('answer') for item in test_data}
    id_to_type = {item['id']: item.get('question_type', QuestionType.DEFAULT) for item in test_data}
    
    # 유형별로 그룹화
    type_predictions = {qt.value: [] for qt in QuestionType}
    type_labels = {qt.value: [] for qt in QuestionType}
    
    all_predictions = []
    all_labels = []
    
    for result in results:
        item_id = result['id']
        pred = result['answer']
        label = id_to_answer.get(item_id)
        q_type = id_to_type.get(item_id, QuestionType.DEFAULT)
        
        if isinstance(q_type, QuestionType):
            q_type = q_type.value
        
        if label is not None:
            type_predictions[q_type].append(pred)
            type_labels[q_type].append(label)
            all_predictions.append(pred)
            all_labels.append(label)
    
    # 유형별 metrics 계산
    metrics_by_type = {}
    for q_type in QuestionType:
        preds = type_predictions[q_type.value]
        labels = type_labels[q_type.value]
        if preds and labels:
            metrics_by_type[q_type.value] = compute_f1_score(preds, labels)
        else:
            metrics_by_type[q_type.value] = {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_weighted': 0.0,
                'correct': 0,
                'total': 0
            }
    
    # 전체 metrics
    metrics_by_type['overall'] = compute_f1_score(all_predictions, all_labels)
    
    return metrics_by_type


def print_evaluation_report(metrics_by_type: Dict[str, Dict[str, float]]) -> None:
    """
    평가 결과를 포맷팅하여 출력합니다.
    
    Args:
        metrics_by_type: compute_f1_by_question_type의 반환값
    """
    print("\n" + "=" * 60)
    print("                    평가 결과 (Evaluation Report)")
    print("=" * 60)
    
    # 유형별 결과
    print("\n[문제 유형별 성능]")
    print("-" * 60)
    print(f"{'유형':<20} {'정확도':>10} {'F1(Macro)':>12} {'정답':>10}")
    print("-" * 60)
    
    for q_type in QuestionType:
        metrics = metrics_by_type.get(q_type.value, {})
        if metrics.get('total', 0) > 0:
            print(f"{q_type.value:<20} {metrics['accuracy']*100:>9.2f}% {metrics['f1_macro']*100:>11.2f}% {metrics['correct']:>5}/{metrics['total']:<4}")
    
    print("-" * 60)
    
    # 전체 결과
    overall = metrics_by_type.get('overall', {})
    if overall.get('total', 0) > 0:
        print(f"\n[전체 성능 (Overall)]")
        print(f"  - Accuracy:     {overall['accuracy']*100:.2f}% ({overall['correct']}/{overall['total']})")
        print(f"  - F1 (Macro):   {overall['f1_macro']*100:.2f}%")
        print(f"  - F1 (Weighted): {overall['f1_weighted']*100:.2f}%")
    
    print("=" * 60 + "\n")


