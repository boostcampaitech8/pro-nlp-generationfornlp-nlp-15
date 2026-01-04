"""
Multi-Agent 추론 스크립트 (80B Verifier 포함)
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
from .prompts import QuestionType, format_question_message, create_messages, SYSTEM_PROMPTS
from .agents.multi_agent_processor import MultiAgentProcessor
from common.utils.wandb import set_wandb_env
from common.utils.logger import setup_logging

# 최대 재시도 횟수
MAX_RETRY = 5


def parse_answer_from_response(response: str, num_choices: int = 5) -> int:
    """
    LLM 응답에서 정답 숫자 추출
    
    Returns:
        추출된 정답 (1~5), 파싱 불가 시 0
    """
    if not response:
        return 0
    
    response = response.strip()
    
    # 패턴 1: "정답: X" 또는 "정답 X" 형태
    pattern1 = r'정답\s*[:\s]\s*(\d)'
    match1 = re.search(pattern1, response)
    if match1:
        answer = int(match1.group(1))
        if 1 <= answer <= num_choices:
            return answer
    
    # 패턴 2: 응답 끝부분의 숫자
    pattern2 = r'(\d)\s*$'
    match2 = re.search(pattern2, response)
    if match2:
        answer = int(match2.group(1))
        if 1 <= answer <= num_choices:
            return answer
    
    # 패턴 3: 응답 전체에서 마지막으로 나타나는 유효 숫자
    valid_numbers = [str(i) for i in range(1, num_choices + 1)]
    for char in reversed(response):
        if char in valid_numbers:
            return int(char)
    
    return 0


class Verifier80BAgent:
    """
    80B 모델을 사용한 Verifier Agent
    """
    def __init__(self, config: dict):
        verifier_config = config.get('verifier_80b', {})
        from openai import AsyncOpenAI
        
        self.client = AsyncOpenAI(
            base_url=verifier_config.get('base_url'),
            api_key=verifier_config.get('api_key', 'EMPTY'),
            timeout=verifier_config.get('timeout', 180),
            max_retries=0,  # 수동 재시도 로직 사용
        )
        self.model_name = verifier_config.get('model_name', 'local_model')
        self.temperature = verifier_config.get('temperature', 0.0)
        self.max_tokens = verifier_config.get('max_tokens', 2048)
        
        self.system_prompt = """당신은 수능/공무원 시험 문제를 검토하는 전문가입니다.
다른 모델의 추론 과정을 검토하고, 오류가 있다면 지적하여 올바른 답을 제시해주세요.

반드시 마지막 줄에 "정답: (숫자)" 형식으로 답을 출력하세요."""
        
        self.user_prompt_template = """[지문]
{paragraph}

[질문]
{question}
{question_plus}

[선택지]
{choices}

[1차 모델의 추론]
{primary_cot}

위 추론에 대해 검토해주세요:
1. 논리적 오류나 사실 오류가 있는지 확인하세요.
2. 누락된 정보나 잘못된 판단이 있는지 확인하세요.
3. 각 선택지에 대한 분석이 올바른지 검토하세요.
4. 최종적으로 올바른 답을 제시하세요.

검토 결과를 상세히 서술하고, 마지막 줄에 "정답: (숫자)" 형식으로 답하세요."""
    
    async def verify(
        self,
        paragraph: str,
        question: str,
        question_plus: str,
        choices: list,
        primary_cot: str,
        semaphore: asyncio.Semaphore,
        max_retries: int = 3,
    ) -> tuple[int, str]:
        """
        80B 모델로 검증합니다.
        
        Returns:
            (answer: int, response: str)
        """
        async with semaphore:
            # 선택지 텍스트 구성
            choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            
            # question_plus 처리
            if question_plus and str(question_plus).strip() not in ['', 'nan', 'None']:
                question_plus_text = f"\n<보기>\n{question_plus}"
            else:
                question_plus_text = ""
            
            # 프롬프트 구성
            user_message = self.user_prompt_template.format(
                paragraph=paragraph,
                question=question,
                question_plus=question_plus_text,
                choices=choices_text,
                primary_cot=primary_cot
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # 재시도 로직
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    
                    content = response.choices[0].message.content
                    answer = parse_answer_from_response(content, len(choices))
                    return answer, content
                    
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)  # 지수 백오프
                        continue
                    else:
                        # 모든 재시도 실패
                        error_msg = f"Error ({type(e).__name__}): {str(e)}"
                        return 0, error_msg
            
            return 0, f"Error: {last_error}"


async def process_single_item_with_80b(
    primary_client: AsyncAPIClient,
    verifier_80b: Verifier80BAgent,
    item: dict,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
    use_type_specific_prompt: bool,
    use_cot: bool,
    verify_all: bool = False,
    verify_threshold: float = None,
) -> dict:
    """
    Multi-Agent 방식으로 단일 문제 처리
    
    Args:
        verify_all: True면 모든 문제를 80B로 검증, False면 조건부
        verify_threshold: 확신도 임계값 (None이면 파싱 실패 시만)
    """
    async with semaphore:
        # 1차 Agent: 기존 CoT 방식
        question_type = item.get('question_type', QuestionType.DEFAULT)
        
        user_message = format_question_message(
            paragraph=item['paragraph'],
            question=item['question'],
            question_plus=item.get('question_plus', ''),
            choices_list=item['choices'],
            question_type=question_type if use_type_specific_prompt else QuestionType.DEFAULT,
            use_cot=use_cot
        )
        
        if use_type_specific_prompt:
            messages = create_messages(user_message, question_type=question_type)
        else:
            messages = create_messages(user_message, system_prompt)
        
        primary_response = ""
        primary_answer = 0
        all_responses = []
        
        # 최대 MAX_RETRY 횟수만큼 시도
        for attempt in range(1, MAX_RETRY + 1):
            try:
                response = await primary_client.chat_completion(messages=messages)
                all_responses.append(f"[Attempt {attempt}] {response}" if response else f"[Attempt {attempt}] NULL")
                
                primary_answer = parse_answer_from_response(response, len(item['choices']))
                
                if primary_answer > 0:
                    primary_response = response
                    break
                    
            except Exception as e:
                all_responses.append(f"[Attempt {attempt}] Error: {str(e)}")
        
        if primary_answer == 0:
            primary_response = " | ".join(all_responses)
        
        # 80B Verifier 호출 여부 결정
        num_choices = len(item['choices'])
        needs_verification = False
        
        if verify_all:
            # 모든 문제 검증
            needs_verification = True
        elif verify_threshold is not None:
            # 확신도 기반 (logit 기반은 추후 구현 가능)
            # 현재는 파싱 실패 시만
            needs_verification = (primary_answer < 1 or primary_answer > num_choices)
        else:
            # 기본: 파싱 실패 시만
            needs_verification = (primary_answer < 1 or primary_answer > num_choices)
        
        # 80B Verifier 호출
        verifier_answer = 0
        verifier_response = ""
        verifier_used = False
        
        if needs_verification:
            verifier_used = True
            verifier_answer, verifier_response = await verifier_80b.verify(
                paragraph=item['paragraph'],
                question=item['question'],
                question_plus=item.get('question_plus', ''),
                choices=item['choices'],
                primary_cot=primary_response,
                semaphore=semaphore,
            )
        
        # 최종 답 결정
        if verifier_used and verifier_answer > 0:
            final_answer = verifier_answer
        else:
            final_answer = primary_answer
        
        # 결과 구성
        question_type_str = question_type.value if hasattr(question_type, 'value') else str(question_type)
        
        # raw_response 구성
        raw_response = f"[Primary Agent]\n{primary_response}"
        if verifier_used:
            raw_response += f"\n\n[80B Verifier]\n{verifier_response}"
        
        return {
            "id": item['id'],
            "answer": str(final_answer) if final_answer > 0 else "0",
            "raw_response": raw_response,
            "question_type": question_type_str,
            "multi_agent_info": {
                "primary_answer": primary_answer,
                "verifier_used": verifier_used,
                "verifier_answer": verifier_answer if verifier_used else None,
            }
        }


async def run_api_inference_agent_async(
    config_path: str,
    mode: str | None = None,
    sample_size: int | None = None,
    input_csv: str | None = None
) -> tuple:
    """
    Multi-Agent 방식으로 수능 문제 추론 실행 (80B Verifier 포함)
    
    Args:
        config_path: inference.yaml 설정 파일 경로
        mode: "train" 또는 "test" (None이면 yaml에서 읽음)
        sample_size: 테스트용 샘플 개수 제한
        input_csv: 기존 CSV 파일 경로 (None이면 yaml에서 읽음, 있으면 80B Verifier만 실행)
    
    Returns:
        (output 파일 경로, raw output 파일 경로) 튜플
    """
    # 설정 로드
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
    
    # 1차 Agent 클라이언트 초기화
    primary_client = AsyncAPIClient(config)
    max_concurrent = config['inference'].get('max_concurrent', 5)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # 80B Verifier 초기화
    if 'verifier_80b' not in config:
        raise ValueError("verifier_80b configuration is required in yaml file")
    
    verifier_80b = Verifier80BAgent(config)
    print(f"\n[80B Verifier] Base URL: {config['verifier_80b'].get('base_url')}")
    
    # 연결 테스트
    try:
        # 간단한 연결 테스트는 생략 (실제 호출 시 에러 처리)
        print("[80B Verifier] Ready")
    except Exception as e:
        logger.warning(f"80B Verifier connection test failed: {e}")
    
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
    
    # 문제 처리
    if use_streaming_save:
        saver = StreamingResultSaver(config['data']['output_dir'], batch_size=save_batch_size)
        
        async def process_and_save(item):
            result = await process_single_item_with_80b(
                primary_client,
                verifier_80b,
                item,
                system_prompt,
                semaphore,
                use_type_specific_prompt,
                use_cot,
                verify_all,
                verify_threshold,
            )
            saver.add_result(result)
            return result
        
        tasks = [process_and_save(item) for item in test_data]
        results = await tqdm_asyncio.gather(*tasks, desc="Multi-Agent Inference")
        
        output_path, raw_output_path = saver.finalize(type_stats)
    else:
        tasks = [
            process_single_item_with_80b(
                primary_client,
                verifier_80b,
                item,
                system_prompt,
                semaphore,
                use_type_specific_prompt,
                use_cot,
                verify_all,
                verify_threshold,
            )
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
    기존 CSV 파일에서 32B CoT 결과를 읽어서 80B Verifier만 실행
    
    Args:
        config_path: inference.yaml 설정 파일 경로
        input_csv: 기존 32B CoT 결과가 있는 CSV 파일 경로
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
    
    # 80B Verifier 초기화
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

