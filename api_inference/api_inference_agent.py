"""
Multi-Agent 추론 스크립트
- 1차 Agent: 기존 CoT 방식으로 문제 풀이
- 2차 Verifier: 80B 모델로 검증 (모든 문제 또는 조건부)
- YAML 설정 기반

사용법:
    python -m api_inference.api_inference_agent --config api_inference/config/inference_agent.yaml
"""
import re
import asyncio
import argparse
import logging
import json
import ast
import pandas as pd
from pathlib import Path
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
from .prompts import QuestionType, SYSTEM_PROMPTS

from .agents.primary_agent import PrimaryAgent
from .agents.verifier_80b_agent import Verifier80BAgent
from .agents.multi_agent_processor_80b import MultiAgentProcessor80B

from common.utils.wandb import set_wandb_env
from common.utils.logger import setup_logging


async def run_api_inference_agent_async(
    config_path: str,
    mode: str | None = None,
    sample_size: int | None = None,
    input_csv: str | None = None
) -> tuple:
    """
    Multi-Agent 방식으로 수능 문제 추론 실행
    
    Args:
        config_path: inference.yaml 설정 파일 경로
        mode: "train" 또는 "test" (None이면 yaml에서 읽음)
        sample_size: 테스트용 샘플 개수 제한
        input_csv: 기존 CSV 파일 경로 (None이면 yaml에서 읽고, 있다면 80B Verifier만 실행)
    
    Returns:
        (output 파일 경로, raw output 파일 경로) 튜플
    """
    ### 1. utils 설정 세팅
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # input_csv 결정: CLI 인자 > yaml 설정
    effective_input_csv = input_csv or config.get('data', {}).get('input_csv')
    
    # CSV 모드: 기존 CSV에서 80B Verifier만 실행
    if effective_input_csv:
        return await run_api_inference_agent_from_csv_async(
            config_path,
            effective_input_csv,
            sample_size or config.get('data', {}).get('sample_size')
        )
    
    # Logger 초기화
    output_dir = Path(config['data']['output_dir'])
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    # httpx, openai 로거 레벨 조정
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    # mode 결정
    if mode is None:
        mode = config.get('mode', 'test')
    
    is_train_mode = (mode == "train")
    logger.info(f"Running in '{mode}' mode (Multi-Agent with 80B Verifier)")
    
    # WandB 초기화
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
                'verifier_80b': config.get('verifier_80b', {}),
            }
        )
        logger.info(f"WandB initialized: {wandb_config.get('project')}/{wandb_config.get('name')}")
    
    ### 2. 추론 LLM 설정 세팅
    # 1차 Agent 클라이언트 초기화
    primary_client = AsyncAPIClient(config)
    max_concurrent = config['inference'].get('max_concurrent', 5)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # 2차 80B Verifier 클라이언트 초기화
    if 'verifier_80b' not in config:
        raise ValueError("verifier_80b configuration is required in yaml file")
    
    verifier_80b = Verifier80BAgent(config)
    print(f"\n[80B Verifier] Base URL: {config['verifier_80b'].get('base_url')}")
    print("[80B Verifier] Ready")
    
    # 데이터 로드
    if is_train_mode:
        data_path = config['data'].get('train_path', 'data/train.csv')
        print(f"[Train Mode] Loading data from: {data_path}")
    else:
        data_path = config['data']['test_path']
        print(f"[Test Mode] Loading data from: {data_path}")
    
    # 시스템 프롬프트
    system_prompt = config['inference'].get('system_prompt')
    if not system_prompt:
        system_prompt = SYSTEM_PROMPTS.get(None)
    
    # 샘플 수 제한
    effective_sample_size = sample_size or config['data'].get('sample_size')
    if effective_sample_size is not None and effective_sample_size > 0:
        print(f"[Sample Mode] Will use {effective_sample_size} samples")
    
    # 유형별 프롬프트 사용 여부
    use_type_specific_prompt = config['inference'].get('use_type_specific_prompt', True)
    
    # 문제 유형 분류
    classification_votes = config['inference'].get('classification_votes', 1)
    
    if use_type_specific_prompt:
        if classification_votes > 1:
            print(f"[Classification] Using LLM classification with {classification_votes} votes")
        else:
            print("[Classification] Using LLM classification")
    else:
        print("[Classification] Skipped (use_type_specific_prompt=false)")
    
    test_data = await load_test_data(
        data_path,
        llm_client=primary_client,
        system_prompt=system_prompt,
        semaphore=semaphore,
        sample_size=effective_sample_size,
        classification_votes=classification_votes,
        skip_classification=not use_type_specific_prompt,
    )
    
    # 문제 유형별 통계
    type_stats = get_question_type_stats(test_data)
    if use_type_specific_prompt:
        print("=== 문제 유형별 통계 ===")
        for qtype, count in type_stats.items():
            print(f"  {qtype}: {count}개")
        print("========================")
    
    # CoT 사용 여부
    use_cot = config['inference'].get('use_cot', False)
    
    # Verifier 설정
    verify_all = config.get('verifier_80b', {}).get('verify_all', False)
    verify_threshold = config.get('verifier_80b', {}).get('logprob_threshold', None)
    
    if verify_all:
        print(f"[80B Verifier] Will verify ALL problems")
    else:
        print(f"[80B Verifier] Will verify only failed parsing (answer not in 1-5)")
    
    # 실시간 저장 모드
    use_streaming_save = config['inference'].get('use_streaming_save', False)
    save_batch_size = config['inference'].get('save_batch_size', 50)
    
    print(f"\nStarting Multi-Agent Inference:")
    print(f"  Primary Agent: CoT={use_cot}, Type-specific={use_type_specific_prompt}")
    print(f"  80B Verifier: verify_all={verify_all}, threshold={verify_threshold}")
    print(f"  Max concurrent: {max_concurrent}")
    
    ### 3. Multi-Agent 설정 세팅
    # Primary Agent 초기화
    primary_agent = PrimaryAgent(
        client=primary_client,
        system_prompt=system_prompt,
        use_type_specific_prompt=use_type_specific_prompt,
        use_cot=use_cot,
    )
    
    # Multi-Agent Processor 초기화
    processor = MultiAgentProcessor80B(
        primary_agent=primary_agent,
        verifier_80b=verifier_80b,
        verify_all=verify_all,
        verify_threshold=verify_threshold,
    )
    
    # 문제 처리
    if use_streaming_save:
        saver = StreamingResultSaver(config['data']['output_dir'], batch_size=save_batch_size)
        
        async def process_and_save(item):
            result = await processor.process_single_item(item, semaphore)
            saver.add_result(result)
            return result
        
        tasks = [process_and_save(item) for item in test_data]
        results = await tqdm_asyncio.gather(*tasks, desc="Multi-Agent Inference")
        
        output_path, raw_output_path = saver.finalize(type_stats)
    else:
        tasks = [
            processor.process_single_item(item, semaphore)
            for item in test_data
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="Multi-Agent Inference")
        
        # ID 순서대로 정렬
        id_order = {item['id']: idx for idx, item in enumerate(test_data)}
        results = sorted(results, key=lambda x: id_order[x['id']])
        
        # 결과 저장
        output_path, raw_output_path = save_results_with_raw(
            results, config['data']['output_dir'], type_stats
        )
    
    # 통계 출력
    failed_count = sum(1 for r in results if r['answer'] == "0")
    verifier_used_count = sum(1 for r in results if r.get('multi_agent_info', {}).get('verifier_used', False))
    
    logger.info(f"Inference Complete! (Failed: {failed_count}/{len(results)}, Verifier used: {verifier_used_count}/{len(results)})")
    print(f"\nInference Complete!")
    print(f"  Failed to parse: {failed_count}/{len(results)}")
    print(f"  80B Verifier used: {verifier_used_count}/{len(results)}")
    
    # Train 모드일 경우 평가
    if is_train_mode:
        metrics_by_type = compute_f1_by_question_type(results, test_data)
        print_evaluation_report(metrics_by_type)
        
        overall = metrics_by_type.get('overall', {})
        logger.info(f"Evaluation Results - Accuracy: {overall.get('accuracy', 0)*100:.2f}%, F1 (Macro): {overall.get('f1_macro', 0)*100:.2f}%")
        
        if wandb_run is not None:
            import wandb
            
            wandb.log({
                'eval/accuracy': overall.get('accuracy', 0),
                'eval/f1_macro': overall.get('f1_macro', 0),
                'eval/f1_weighted': overall.get('f1_weighted', 0),
                'eval/correct': overall.get('correct', 0),
                'eval/total': overall.get('total', 0),
                'eval/failed_parse': failed_count,
                'eval/verifier_used': verifier_used_count,
            })
            
            logger.info("Metrics logged to WandB")
    
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Raw responses saved to: {raw_output_path}")
    print(f"  - Results saved to: {output_path}")
    print(f"  - Raw responses saved to: {raw_output_path}")
    
    if wandb_run is not None:
        wandb_run.finish()
        logger.info("WandB run finished")
    
    return output_path, raw_output_path


def parse_problems_from_csv(problems_str: str) -> dict:
    """CSV의 problems 컬럼을 파싱"""
    if pd.isna(problems_str) or not problems_str:
        return {}
    
    try:
        return ast.literal_eval(problems_str)
    except (ValueError, SyntaxError):
        try:
            problems_str = str(problems_str).replace("'", '"')
            return json.loads(problems_str)
        except:
            return {}


async def run_api_inference_agent_from_csv_async(
    config_path: str,
    input_csv: str,
    sample_size: int | None = None
) -> tuple:
    """
    기존 CSV 파일에서 CoT 결과를 읽어서 80B Verifier만 실행
    
    Args:
        config_path: inference.yaml 설정 파일 경로
        input_csv: 기존 CoT 결과가 있는 CSV 파일 경로
        sample_size: 테스트용 샘플 개수 제한
    
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
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    logger.info(f"Running 80B Verifier only mode from CSV: {input_csv}")
    print(f"\n[Mode] 80B Verifier Only (from existing CSV)")
    print(f"  Input CSV: {input_csv}")
    
    # 80B Verifier 클라이언트 초기화
    if 'verifier_80b' not in config:
        raise ValueError("verifier_80b configuration is required in yaml file")
    
    verifier_80b = Verifier80BAgent(config)
    print(f"\n[80B Verifier] Base URL: {config['verifier_80b'].get('base_url')}")
    
    # 세마포어 설정 (80B는 느리므로 더 낮게 설정)
    max_concurrent_verifier = config.get('verifier_80b', {}).get('max_concurrent', 3)
    semaphore = asyncio.Semaphore(max_concurrent_verifier)
    print(f"  Max concurrent: {max_concurrent_verifier}")
    
    # Verifier 설정 확인
    verify_all = config.get('verifier_80b', {}).get('verify_all', False)
    verify_threshold = config.get('verifier_80b', {}).get('logprob_threshold', None)
    
    if verify_all:
        print(f"[80B Verifier] Will verify ALL problems")
    else:
        print(f"[80B Verifier] Will verify all problems (CSV mode)")
    
    # CSV 읽기
    print(f"\nLoading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if sample_size:
        df = df.head(sample_size)
        print(f"  Using sample: {sample_size} rows")
    
    print(f"  Total rows: {len(df)}")
    
    # 필수 컬럼 확인
    required_cols = ['id', 'raw_response', 'paragraph']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # problems 컬럼이 있으면 파싱, 없으면 choices 컬럼 사용
    has_problems = 'problems' in df.columns
    
    # 각 행 처리
    async def process_row(row: pd.Series) -> dict:
        # problems 파싱 또는 choices 사용
        if has_problems:
            problems = parse_problems_from_csv(row.get('problems', ''))
            question = problems.get('question', '')
            choices_list = problems.get('choices', [])
        else:
            # choices 컬럼이 있을 수도 있음 (또는 문제 정보 추출 필요)
            # 여기서는 간단히 처리
            question = row.get('question', '')
            if 'choices' in df.columns:
                choices_str = row.get('choices', '')
                try:
                    choices_list = ast.literal_eval(choices_str) if isinstance(choices_str, str) else choices_str
                except:
                    choices_list = []
            else:
                choices_list = []
        
        if not choices_list:
            return {
                "id": row['id'],
                "answer": "0",
                "raw_response": "[80B Verifier] Error: Failed to parse choices",
                "question_type": row.get('question_type', 'default'),
            }
        
        # 32B의 CoT 가져오기
        primary_cot = row.get('raw_response', '')
        if not primary_cot or pd.isna(primary_cot):
            primary_cot = "No CoT available"
        
        # question_plus 처리
        question_plus = row.get('question_plus', '')
        
        # 80B Verifier 호출
        verifier_answer, verifier_response = await verifier_80b.verify(
            paragraph=row['paragraph'],
            question=question,
            question_plus=question_plus,
            choices=choices_list,
            primary_cot=primary_cot,
            semaphore=semaphore,
        )
        
        # 결과 구성
        raw_response = f"[32B Primary CoT]\n{primary_cot}\n\n[80B Verifier]\n{verifier_response}"
        
        return {
            "id": row['id'],
            "answer": str(verifier_answer) if verifier_answer > 0 else "0",
            "raw_response": raw_response,
            "question_type": row.get('question_type', 'default'),
            "multi_agent_info": {
                "primary_answer": int(row.get('llm_pred', 0)) if 'llm_pred' in row else 0,
                "verifier_used": True,
                "verifier_answer": verifier_answer,
            }
        }
    
    # 병렬 처리
    tasks = [process_row(row) for _, row in df.iterrows()]
    results = await tqdm_asyncio.gather(*tasks, desc="80B Verification")
    
    # ID 순서대로 정렬
    id_order = {row['id']: idx for idx, row in df.iterrows()}
    results = sorted(results, key=lambda x: id_order[x['id']])
    
    # 문제 유형별 통계 (results에서 question_type 추출)
    type_stats = {}
    for result in results:
        qtype = result.get("question_type", "default")
        if isinstance(qtype, QuestionType):
            qtype = qtype.value
        type_stats[qtype] = type_stats.get(qtype, 0) + 1
    
    # 결과 저장
    output_path, raw_output_path = save_results_with_raw(
        results, config['data']['output_dir'], type_stats
    )
    
    # 통계 출력
    failed_count = sum(1 for r in results if r['answer'] == "0")
    valid_count = len(results) - failed_count
    
    logger.info(f"Verification Complete! (Valid: {valid_count}/{len(results)}, Failed: {failed_count}/{len(results)})")
    print(f"\nVerification Complete!")
    print(f"  Valid answers (1-5): {valid_count}/{len(results)} ({valid_count/len(results)*100:.1f}%)")
    print(f"  Failed to parse: {failed_count}/{len(results)} ({failed_count/len(results)*100:.1f}%)")
    
    # Train 데이터와 비교 (answer 컬럼이 있으면)
    if 'answer' in df.columns:
        correct_32b = (df['llm_pred'].astype(str) == df['answer'].astype(str)).sum()
        correct_80b = sum(1 for r, a in zip(results, df['answer'].astype(str)) if r['answer'] == str(a))
        
        print(f"\n=== Accuracy Comparison ===")
        print(f"  32B (original): {correct_32b}/{len(df)} ({correct_32b/len(df)*100:.1f}%)")
        print(f"  80B (verifier): {correct_80b}/{len(df)} ({correct_80b/len(df)*100:.1f}%)")
        print(f"  Improvement: {correct_80b - correct_32b} problems")
    
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Raw responses saved to: {raw_output_path}")
    print(f"\n  - Results saved to: {output_path}")
    print(f"  - Raw responses saved to: {raw_output_path}")
    
    return output_path, raw_output_path


def run_api_inference_agent(
    config_path: str,
    mode: str | None = None,
    sample_size: int | None = None,
    input_csv: str | None = None
) -> tuple:
    """
    동기 래퍼 함수
    
    Args:
        input_csv: 기존 CSV 파일 경로 (None이면 yaml에서 읽음)
    """
    return asyncio.run(run_api_inference_agent_async(config_path, mode, sample_size, input_csv))


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent 추론 (80B Verifier 포함)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="inference.yaml 설정 파일 경로"
    )
    
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Train 모드: train.csv로 추론 후 F1 Score 계산 및 wandb 기록"
    )
    mode_group.add_argument(
        "--test",
        action="store_true",
        help="Test 모드: test.csv로 추론 후 결과만 저장"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="테스트용 샘플 개수 제한 (예: --sample 100)"
    )
    
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="기존 32B CoT 결과 CSV 파일 경로 (yaml 설정 오버라이드, 예: --input-csv data/output.csv)"
    )
    
    args = parser.parse_args()
    
    mode = None
    if args.train:
        mode = "train"
    elif args.test:
        mode = "test"
    
    run_api_inference_agent(
        args.config,
        mode=mode,
        sample_size=args.sample,
        input_csv=args.input_csv
    )


if __name__ == "__main__":
    main()

