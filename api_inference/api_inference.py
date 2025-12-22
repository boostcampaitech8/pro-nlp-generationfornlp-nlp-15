"""
일반 응답 파싱 기반 수능 문제 추론 스크립트
- LLM 응답의 마지막 부분에서 숫자(1-5) 추출
- 정답: 1-indexed (1, 2, 3, 4, 5)
- 비동기 병렬 처리 지원
- 파싱 실패 시 자동 재시도 (max 5회)

사용법:
    python -m api_inference.api_inference --config api_inference/config/inference.yaml
"""
import re
import asyncio
import argparse
import yaml
from tqdm.asyncio import tqdm_asyncio

from .utils.api_client import AsyncAPIClient
from .utils.data_loader import (
    load_test_data, format_question_message, create_messages,
    QuestionType, SYSTEM_PROMPTS, get_question_type_stats
)
from .utils.output_handler import save_results_with_raw

# 최대 재시도 횟수
MAX_RETRY = 5


def parse_answer_from_response(response: str, num_choices: int = 5) -> str:
    """
    LLM 응답에서 정답 숫자 추출
    응답의 마지막 부분에서 유효한 숫자(1-5 또는 1-4)를 찾음
    
    Args:
        response: LLM 응답 텍스트
        num_choices: 선택지 개수 (4 또는 5)
    
    Returns:
        추출된 정답 문자열 ("1" ~ "5"), 파싱 불가 시 "0"
    """
    if not response:
        return "0"  # 파싱 불가
    
    # 응답 정리
    response = response.strip()
    
    # 패턴 1: "정답: X" 또는 "정답 X" 형태
    pattern1 = r'정답\s*[:\s]\s*(\d)'
    match1 = re.search(pattern1, response)
    if match1:
        answer = match1.group(1)
        if 1 <= int(answer) <= num_choices:
            return answer
    
    # 패턴 2: 응답 끝부분의 숫자
    # 마지막에서 찾기 (역순으로)
    pattern2 = r'(\d)\s*$'
    match2 = re.search(pattern2, response)
    if match2:
        answer = match2.group(1)
        if 1 <= int(answer) <= num_choices:
            return answer
    
    # 패턴 3: 응답 전체에서 마지막으로 나타나는 유효 숫자
    valid_numbers = [str(i) for i in range(1, num_choices + 1)]
    for char in reversed(response):
        if char in valid_numbers:
            return char
    
    # 패턴 4: 괄호 안의 숫자 (예: (1), ①)
    pattern4 = r'[(\[【]?\s*(\d)\s*[)\]】]?'
    matches = re.findall(pattern4, response)
    for match in reversed(matches):
        if 1 <= int(match) <= num_choices:
            return match
    
    # 찾지 못한 경우 파싱 불가
    return "0"


async def process_single_item(
    client: AsyncAPIClient,
    item: dict,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
    use_type_specific_prompt: bool = True
) -> dict:
    """
    단일 문제 처리 (비동기) - 파싱 실패 시 자동 재시도
    
    Args:
        client: 비동기 API 클라이언트
        item: 테스트 데이터 항목
        system_prompt: 시스템 프롬프트 (use_type_specific_prompt=False일 때 사용)
        semaphore: 동시 요청 제한용 세마포어
        use_type_specific_prompt: 문제 유형별 프롬프트 사용 여부
    
    Returns:
        결과 딕셔너리 {'id': ..., 'answer': ..., 'raw_response': ..., 'question_type': ...}
    """
    async with semaphore:
        # 문제 유형 가져오기
        question_type = item.get('question_type', QuestionType.DEFAULT)
        
        # 프롬프트 생성 (유형별 또는 기본)
        user_message = format_question_message(
            paragraph=item['paragraph'],
            question=item['question'],
            question_plus=item['question_plus'],
            choices_list=item['choices'],
            question_type=question_type if use_type_specific_prompt else QuestionType.DEFAULT
        )
        
        # 메시지 구성 (유형별 시스템 프롬프트 또는 기본)
        if use_type_specific_prompt:
            messages = create_messages(user_message, question_type=question_type)
        else:
            messages = create_messages(user_message, system_prompt)
        
        raw_response = ""
        answer = "0"
        all_responses = []  # 모든 시도의 응답 저장
        
        # 최대 MAX_RETRY 횟수만큼 시도
        for attempt in range(1, MAX_RETRY + 1):
            try:
                # 일반 채팅 요청
                response = await client.chat_completion(messages=messages)
                all_responses.append(f"[Attempt {attempt}] {response}" if response else f"[Attempt {attempt}] NULL")
                
                # 응답에서 정답 파싱
                answer = parse_answer_from_response(response, item['num_choices'])
                
                # 파싱 성공 시 루프 종료
                if answer != "0":
                    raw_response = response
                    break
                    
            except Exception as e:
                all_responses.append(f"[Attempt {attempt}] Error: {str(e)}")
        
        # 모든 시도 후에도 파싱 실패한 경우
        if answer == "0":
            raw_response = " | ".join(all_responses)
        
        return {
            "id": item['id'],
            "answer": answer,
            "raw_response": raw_response,
            "question_type": question_type.value
        }


async def run_api_inference_async(config_path: str) -> tuple:
    """
    일반 응답 파싱 방식으로 수능 문제 추론 실행 (비동기 병렬)
    
    Args:
        config_path: inference.yaml 설정 파일 경로
    
    Returns:
        (output 파일 경로, raw output 파일 경로) 튜플
    """
    # 설정 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 비동기 API 클라이언트 초기화
    client = AsyncAPIClient(config)
    
    # 테스트 데이터 로드
    test_data = load_test_data(config['data']['test_path'])
    
    # 문제 유형별 통계 출력
    type_stats = get_question_type_stats(test_data)
    print("=== 문제 유형별 통계 ===")
    for qtype, count in type_stats.items():
        print(f"  {qtype}: {count}개")
    print("========================")
    
    # 유형별 프롬프트 사용 여부
    use_type_specific_prompt = config['inference'].get('use_type_specific_prompt', True)
    
    # 시스템 프롬프트 (유형별 프롬프트 미사용 시에만 적용)
    system_prompt = config['inference'].get(
        'system_prompt'
    )
    
    # 동시 요청 수 설정
    max_concurrent = config['inference'].get('max_concurrent', 40)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    print(f"Starting API Inference (max_concurrent={max_concurrent}, max_retry={MAX_RETRY}, use_type_specific_prompt={use_type_specific_prompt})...")
    
    # 모든 태스크 생성
    tasks = [
        process_single_item(client, item, system_prompt, semaphore, use_type_specific_prompt)
        for item in test_data
    ]
    
    # 병렬 실행 with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc="Inference")
    
    # ID 순서대로 정렬 (원래 순서 유지)
    id_order = {item['id']: idx for idx, item in enumerate(test_data)}
    results = sorted(results, key=lambda x: id_order[x['id']])
    
    # 파싱 실패 통계
    failed_count = sum(1 for r in results if r['answer'] == "0")
    print(f"Inference Complete! (Failed to parse: {failed_count}/{len(results)})")
    
    # 결과 저장 (output.csv + output_raw.csv)
    output_path, raw_output_path = save_results_with_raw(results, config['data']['output_dir'])
    print(f"  - Results saved to: {output_path}")
    print(f"  - Raw responses saved to: {raw_output_path}")
    
    return output_path, raw_output_path


def run_api_inference(config_path: str) -> tuple:
    """동기 래퍼 함수"""
    return asyncio.run(run_api_inference_async(config_path))


def main():
    parser = argparse.ArgumentParser(
        description="일반 응답 파싱 기반 수능 문제 추론"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="inference.yaml 설정 파일 경로"
    )
    args = parser.parse_args()
    
    run_api_inference(args.config)


if __name__ == "__main__":
    main()
