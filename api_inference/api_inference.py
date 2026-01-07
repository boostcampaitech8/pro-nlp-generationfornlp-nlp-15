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
import logging
from pathlib import Path
from typing import Any
import yaml
from tqdm.asyncio import tqdm_asyncio

from .utils.api_client import AsyncAPIClient
from .utils.data_loader import load_test_data
from .utils.metrics import (
    get_question_type_stats,
    compute_f1_by_question_type,
    print_evaluation_report,
)
from .utils.output_handler import save_results_with_raw, StreamingResultSaver

from common.utils.wandb import set_wandb_env
from common.utils.logger import setup_logging

from .prompts import QuestionType, format_question_message, create_messages
from .agents import MultiAgentProcessor

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
    use_type_specific_prompt: bool = True,
    use_cot: bool = False
) -> dict:
    """
    단일 문제 처리 (비동기) - 파싱 실패 시 자동 재시도
    
    Args:
        client: 비동기 API 클라이언트
        item: 테스트 데이터 항목
        system_prompt: 시스템 프롬프트 (use_type_specific_prompt=False일 때 사용)
        semaphore: 동시 요청 제한용 세마포어
        use_type_specific_prompt: 문제 유형별 프롬프트 사용 여부
        use_cot: CoT 프롬프트 사용 여부
    
    Returns:
        결과 딕셔너리 {'id': ..., 'answer': ..., 'raw_response': ..., 'question_type': ...}
    """
    async with semaphore:
        # 문제 유형 가져오기
        question_type = item.get('question_type', QuestionType.DEFAULT)
        
        # 프롬프트 생성 (유형별/기본 프롬프트, CoT 여부)
        user_message = format_question_message(
            paragraph=item['paragraph'],
            question=item['question'],
            question_plus=item['question_plus'],
            choices_list=item['choices'],
            question_type=question_type if use_type_specific_prompt else QuestionType.DEFAULT,
            use_cot=use_cot
        )
        
        # 유형별 시스템 프롬프트 또는 기본 프롬프트
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


async def run_api_inference_async(config_path: str, mode: str | None = None, sample_size: int | None = None) -> tuple:
    """
    일반 응답 파싱 방식으로 수능 문제 추론 실행 (비동기 병렬)
    
    Args:
        config_path: inference.yaml 설정 파일 경로
        mode: "train" 또는 "test" (None이면 yaml에서 읽음)
              - train: train.csv 사용, F1 Score 계산 및 wandb 기록
              - test: test.csv 사용, 결과만 저장
        sample_size: 테스트용 샘플 개수 제한 (None이면 전체 데이터 사용)
    
    Returns:
        (output 파일 경로, raw output 파일 경로) 튜플
    """
    # 설정 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Logger 초기화
    output_dir = Path(config['data']['output_dir'])
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    # 불필요한 HTTP 로그 숨김
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    # mode 결정: 기본값 "test"
    if mode is None:
        mode = config.get('mode', 'test')
    
    is_train_mode = (mode == "train")
    logger.info(f"Running in '{mode}' mode")
    
    # WandB 초기화 (train 모드일 때만)
    wandb_run = None
    if is_train_mode and 'wandb' in config:
        import wandb
        wandb_config = config['wandb']
        set_wandb_env(
            project=wandb_config.get('project'),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('name'),
        )
        wandb_run = wandb.init(
            project=wandb_config.get('project'),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('name'),
            config={
                'api': config.get('api', {}),
                'inference': config.get('inference', {}),
                'data': config.get('data', {}),
            }
        )
        logger.info(f"WandB initialized: {wandb_config.get('project')}/{wandb_config.get('name')}")
    
    # 비동기 API 클라이언트 초기화
    client = AsyncAPIClient(config)
    # 동시 요청 수 설정
    max_concurrent = config['inference'].get('max_concurrent', 40)
    semaphore = asyncio.Semaphore(max_concurrent)

    # 데이터 로드 (train 모드면 train_path, test 모드면 test_path)
    if is_train_mode:
        data_path = config['data'].get('train_path', 'data/train.csv')
        print(f"[Train Mode] Loading data from: {data_path}")
    else:
        data_path = config['data']['test_path']
        print(f"[Test Mode] Loading data from: {data_path}")
    
    # 시스템 프롬프트 (유형별 프롬프트 미사용 시에만 적용)
    system_prompt = config['inference'].get('system_prompt')
    if not system_prompt:
        from .prompts import SYSTEM_PROMPTS
        system_prompt = SYSTEM_PROMPTS.get(None)

    # 데이터셋 sample size 만큼 subset 사용할건지
    effective_sample_size = sample_size or config['data'].get('sample_size')
    if effective_sample_size is not None and effective_sample_size > 0:
        print(f"[Sample Mode] Will use {effective_sample_size} samples")

    # 유형별 프롬프트 사용 여부
    use_type_specific_prompt = config['inference'].get('use_type_specific_prompt', True)
    
    # 문제 유형 분류 투표 횟수 (다수결 방식)
    classification_votes = config['inference'].get('classification_votes', 1)
    
    # use_type_specific_prompt가 false면 LLM 분류 스킵
    if use_type_specific_prompt:
        if classification_votes > 1:
            print(f"[Classification] Using LLM classification with {classification_votes} votes")
        else:
            print("[Classification] Using LLM classification")
    else:
        print("[Classification] Skipped (use_type_specific_prompt=false)")

    test_data = await load_test_data(
        data_path,
        llm_client=client,
        system_prompt=system_prompt,
        semaphore=semaphore,
        sample_size=effective_sample_size,
        classification_votes=classification_votes,
        skip_classification=not use_type_specific_prompt,  # false면 분류 스킵
    )
    
    # 문제 유형별 통계 출력
    type_stats = get_question_type_stats(test_data)
    if use_type_specific_prompt:
        print("=== 문제 유형별 통계 ===")
        for qtype, count in type_stats.items():
            print(f"  {qtype}: {count}개")
        print("========================")
    
    # 시스템 프롬프트
    system_prompt = config['inference'].get(
        'system_prompt'
    )
    
    # CoT (Chain of Thought) 프롬프트 사용 여부
    use_cot = config['inference'].get('use_cot', False)
    
    # Multi-Agent 모드 사용 여부 (현재는 검증 Agent 사용 여부)
    use_multi_agent = config['inference'].get('use_multi_agent', False)
    
    # 배치 저장 크기 (기본값: 50)
    save_batch_size = config['inference'].get('save_batch_size', 50)
    
    # 실시간 저장 모드 사용 여부
    use_streaming_save = config['inference'].get('use_streaming_save', True)
    
    if use_multi_agent:
        print(f"Starting Multi-Agent Inference (max_concurrent={max_concurrent}, Primary CoT + Verifier)...")
        print(f"  - use_type_specific_prompt: {use_type_specific_prompt}")
        print(f"  - use_cot: {use_cot}")
        
        # Multi-Agent 프로세서 초기화 (기존 CoT 방식 + Verifier)
        multi_agent_processor = MultiAgentProcessor(
            client=client,
            system_prompt=system_prompt,
            use_type_specific_prompt=use_type_specific_prompt,
            use_cot=use_cot,
        )
        
        # asyncio.gather 방식으로 처리
        tasks = [
            multi_agent_processor.process_single_item(item, semaphore)
            for item in test_data
        ]
        multi_agent_results = await tqdm_asyncio.gather(*tasks, desc="Multi-Agent Inference")
        
        # 결과 변환
        results = [r.to_output_dict() for r in multi_agent_results]
        
        # ID 순서대로 정렬
        id_order = {item['id']: idx for idx, item in enumerate[dict[str, Any]](test_data)}
        results = sorted(results, key=lambda x: id_order[x['id']])
        
        # 결과 저장
        output_path, raw_output_path = save_results_with_raw(
            results, config['data']['output_dir'], type_stats
        )
        
        # Multi-Agent 통계 출력
        verifier_used_count = sum(1 for r in multi_agent_results if r.verifier_used)
        print(f"[Multi-Agent Stats] Verifier used: {verifier_used_count}/{len(results)}")
    
    elif use_streaming_save:
        print(f"Starting API Inference (max_concurrent={max_concurrent}, max_retry={MAX_RETRY}, use_type_specific_prompt={use_type_specific_prompt}, use_cot={use_cot}, streaming_save={use_streaming_save})...")
        
        # 실시간/배치 저장 모드
        saver = StreamingResultSaver(config['data']['output_dir'], batch_size=save_batch_size)
        
        async def process_and_save(item):
            result = await process_single_item(client, item, system_prompt, semaphore, use_type_specific_prompt, use_cot)
            saver.add_result(result)
            return result
        
        tasks = [process_and_save(item) for item in test_data]
        results = await tqdm_asyncio.gather(*tasks, desc="Inference")
        
        # 남은 결과 저장 및 최종화
        output_path, raw_output_path = saver.finalize(type_stats)
    else:
        print(f"Starting API Inference (max_concurrent={max_concurrent}, max_retry={MAX_RETRY}, use_type_specific_prompt={use_type_specific_prompt}, use_cot={use_cot}, streaming_save={use_streaming_save})...")
        
        # 기존 방식: 한 번에 저장
        tasks = [
            process_single_item(client, item, system_prompt, semaphore, use_type_specific_prompt, use_cot)
            for item in test_data
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="Inference")
        
        # ID 순서대로 정렬 (원래 순서 유지)
        id_order = {item['id']: idx for idx, item in enumerate(test_data)}
        results = sorted(results, key=lambda x: id_order[x['id']])
        
        # 결과 저장
        output_path, raw_output_path = save_results_with_raw(
            results, config['data']['output_dir'], type_stats
        )
    
    # 파싱 실패 통계
    failed_count = sum(1 for r in results if r['answer'] == "0")
    logger.info(f"Inference Complete! (Failed to parse: {failed_count}/{len(results)})")
    print(f"Inference Complete! (Failed to parse: {failed_count}/{len(results)})")
    
    # Train 모드일 경우 F1 Score 계산 및 출력
    if is_train_mode:
        metrics_by_type = compute_f1_by_question_type(results, test_data)
        print_evaluation_report(metrics_by_type)
        
        # Logger에 기록
        overall = metrics_by_type.get('overall', {})
        logger.info(f"Evaluation Results - Accuracy: {overall.get('accuracy', 0)*100:.2f}%, F1 (Macro): {overall.get('f1_macro', 0)*100:.2f}%")
        
        # WandB에 기록
        if wandb_run is not None:
            wandb.log({
                'eval/accuracy': overall.get('accuracy', 0),
                'eval/f1_macro': overall.get('f1_macro', 0),
                'eval/f1_weighted': overall.get('f1_weighted', 0),
                'eval/correct': overall.get('correct', 0),
                'eval/total': overall.get('total', 0),
                'eval/failed_parse': failed_count,
            })
            
            # 문제 유형별 metrics
            for q_type in QuestionType:
                type_metrics = metrics_by_type.get(q_type.value, {})
                if type_metrics.get('total', 0) > 0:
                    wandb.log({
                        f'eval/{q_type.value}/accuracy': type_metrics.get('accuracy', 0),
                        f'eval/{q_type.value}/f1_macro': type_metrics.get('f1_macro', 0),
                        f'eval/{q_type.value}/correct': type_metrics.get('correct', 0),
                        f'eval/{q_type.value}/total': type_metrics.get('total', 0),
                    })
            
            logger.info("Metrics logged to WandB")
    
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Raw responses saved to: {raw_output_path}")
    print(f"  - Results saved to: {output_path}")
    print(f"  - Raw responses saved to: {raw_output_path}")
    
    # WandB 종료
    if wandb_run is not None:
        wandb_run.finish()
        logger.info("WandB run finished")
    
    return output_path, raw_output_path


def run_api_inference(config_path: str, mode: str | None = None, sample_size: int | None = None) -> tuple:
    """
    동기 래퍼 함수
    
    Args:
        config_path: 설정 파일 경로
        mode: "train" 또는 "test" (None이면 yaml에서 읽음)
        sample_size: 테스트용 샘플 개수 제한
    """
    return asyncio.run(run_api_inference_async(config_path, mode, sample_size))


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
    
    # --train / --test 선택
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Train 모드: train.csv로 추론 후 F1 Score 계산 및 wandb 기록 (yaml 설정 오버라이드)"
    )
    mode_group.add_argument(
        "--test",
        action="store_true",
        help="Test 모드: test.csv로 추론 후 결과만 저장 (yaml 설정 오버라이드)"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="테스트용 샘플 개수 제한 (예: --sample 100)"
    )
    
    args = parser.parse_args()
    
    mode = None
    if args.train:
        mode = "train"
    elif args.test:
        mode = "test"
    
    run_api_inference(args.config, mode, sample_size=args.sample)


if __name__ == "__main__":
    main()
