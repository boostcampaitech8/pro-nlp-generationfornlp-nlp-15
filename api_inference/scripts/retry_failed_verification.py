"""
실패한 검증 재시도 스크립트

결과 파일에서 정답이 0 (파싱 실패)인 문제들만 골라서
80B Verifier로 재검증합니다.

사용법:
    python -m api_inference.scripts.retry_failed_verification \
        --config api_inference/config/inference_agent.yaml \
        --result-csv data/agent_80b_train/output_raw_7564439.csv \
        --source-csv data/train_cot_logit_both_fail.csv \
        --output-csv data/agent_80b_train/output_raw_retry.csv
"""
import asyncio
import argparse
import json
import ast
import pandas as pd
from pathlib import Path
import yaml
import logging
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

logging.basicConfig(level=logging.WARNING)  # WARNING 이상만 출력
logger = logging.getLogger(__name__)

from api_inference.api_inference_agent import Verifier80BAgent, parse_answer_from_response


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


async def retry_verification_single(
    row: pd.Series,
    verifier_80b: Verifier80BAgent,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    단일 문제 재검증
    
    Returns:
        {"id": ..., "answer": ..., "raw_response": ...}
    """
    problem_id = row['id']
    
    try:
        # problems 파싱 (semaphore 밖에서 처리)
        problems = parse_problems_from_csv(row.get('problems', ''))
        question = problems.get('question', '')
        choices_list = problems.get('choices', [])
        
        if not question or not choices_list:
            logger.warning(f"{problem_id}: Failed to parse problems")
            return {
                "id": problem_id,
                "answer": "0",
                "raw_response": "Error: Failed to parse problems column",
                "question_type": row.get('question_type', 'default'),
            }
        
        # 32B의 CoT 가져오기
        primary_cot = row.get('raw_response', '')
        if not primary_cot or pd.isna(primary_cot):
            primary_cot = "No CoT available"
        
        # question_plus 처리
        question_plus = row.get('question_plus', '')
        
        # 80B Verifier 호출
        # verify 함수 내부에서 semaphore를 사용하므로 여기서는 전달만 함
        verifier_answer, verifier_response = await verifier_80b.verify(
            paragraph=row['paragraph'],
            question=question,
            question_plus=question_plus,
            choices=choices_list,
            primary_cot=primary_cot,
            semaphore=semaphore,  # verify 함수가 내부적으로 사용
            max_retries=3,
        )
        
        # 결과 구성
        raw_response = f"[32B Primary CoT]\n{primary_cot}\n\n[80B Verifier - Retry]\n{verifier_response}"
        
        return {
            "id": problem_id,
            "answer": str(verifier_answer) if verifier_answer > 0 else "0",
            "raw_response": raw_response,
            "question_type": row.get('question_type', 'default'),
        }
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(f"{problem_id}: {error_type} - {error_msg}")
        
        return {
            "id": problem_id,
            "answer": "0",
            "raw_response": f"Retry Error ({error_type}): {error_msg}",
            "question_type": row.get('question_type', 'default'),
        }


async def main():
    parser = argparse.ArgumentParser(
        description="실패한 검증만 재시도 (정답이 0인 문제들)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="inference.yaml 설정 파일 경로"
    )
    parser.add_argument(
        "--result-csv",
        type=str,
        required=True,
        help="검증 결과 CSV 파일 경로 (output_raw_*.csv)"
    )
    parser.add_argument(
        "--source-csv",
        type=str,
        required=True,
        help="원본 CSV 파일 경로 (32B CoT가 있는 파일)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="재검증 결과 저장 CSV 파일 경로"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="동시 요청 수 (기본값: 2, 80B는 느리므로 낮게 설정)"
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 결과 CSV 읽기
    result_df = pd.read_csv(args.result_csv)
    
    # 정답이 0인 문제들 찾기
    failed_ids = result_df[result_df['answer'].astype(str) == '0']['id'].tolist()
    
    if len(failed_ids) == 0:
        return
    
    # 원본 CSV 읽기
    source_df = pd.read_csv(args.source_csv)
    
    # 실패한 문제들만 필터링
    failed_df = source_df[source_df['id'].isin(failed_ids)].copy()
    
    if len(failed_df) == 0:
        return
    
    # 80B Verifier 초기화
    if 'verifier_80b' not in config:
        raise ValueError("verifier_80b configuration is required in yaml file")
    
    verifier_80b = Verifier80BAgent(config)
    
    # 세마포어 설정
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    # 모든 작업 생성
    tasks = [
        retry_verification_single(row, verifier_80b, semaphore)
        for _, row in failed_df.iterrows()
    ]
    
    # tqdm으로 진행률 표시하면서 gather 실행
    pbar = tqdm(total=len(tasks), desc="Retry Verification")
    
    # 진행률 업데이트를 위한 태스크 래퍼
    async def update_progress_wrapper(coro):
        try:
            result = await coro
            pbar.update(1)
            return result
        except Exception as e:
            pbar.update(1)
            return e
    
    # 모든 태스크를 진행률 업데이트와 함께 실행
    wrapped_tasks = [update_progress_wrapper(task) for task in tasks]
    
    retry_results = await asyncio.gather(*wrapped_tasks, return_exceptions=True)
    pbar.close()
    
    # 예외 처리된 결과들 확인
    exceptions = [r for r in retry_results if isinstance(r, Exception)]
    
    # 예외를 결과 딕셔너리로 변환
    processed_results = []
    for i, result in enumerate(retry_results):
        if isinstance(result, Exception):
            # 예외 발생 시 기본 결과 반환
            problem_id = failed_df.iloc[i]['id']
            processed_results.append({
                "id": problem_id,
                "answer": "0",
                "raw_response": f"Exception: {type(result).__name__}: {str(result)}",
                "question_type": failed_df.iloc[i].get('question_type', 'default'),
            })
        else:
            processed_results.append(result)
    
    retry_results = processed_results
    
    # 결과를 딕셔너리로 변환
    retry_dict = {r['id']: r for r in retry_results}
    
    # answer 컬럼을 문자열 타입으로 변환 (타입 불일치 경고 방지)
    if 'answer' in result_df.columns:
        result_df['answer'] = result_df['answer'].astype(str)
    
    # 기존 결과 업데이트
    for idx, row in result_df.iterrows():
        if row['id'] in retry_dict:
            retry_result = retry_dict[row['id']]
            result_df.at[idx, 'answer'] = str(retry_result['answer'])
            result_df.at[idx, 'raw_response'] = retry_result['raw_response']
            # question_type도 업데이트 (없으면 추가)
            if 'question_type' in result_df.columns:
                result_df.at[idx, 'question_type'] = retry_result['question_type']
    
    # 결과 저장
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 재검증된 문제들만 별도로 저장 (디버깅/확인용)
    retried_only_df = result_df[result_df['id'].isin(failed_ids)].copy()
    retried_only_path = output_path.parent / f"{output_path.stem}_retried_only.csv"
    retried_only_df.to_csv(retried_only_path, index=False, encoding='utf-8-sig')
    
    # 재검증된 문제들만 간단 버전 저장
    retried_simple_path = output_path.parent / f"output_{output_path.stem.replace('raw_', '')}_retried_only.csv"
    retried_only_df[['id', 'answer']].to_csv(retried_simple_path, index=False)
    
    # output.csv 저장 (id, answer만) - 전체 파일
    output_simple_path = output_path.parent / f"output_{output_path.stem.replace('raw_', '')}.csv"
    result_df[['id', 'answer']].to_csv(output_simple_path, index=False)
    
    # output_raw.csv 저장 - 전체 파일
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    asyncio.run(main())

