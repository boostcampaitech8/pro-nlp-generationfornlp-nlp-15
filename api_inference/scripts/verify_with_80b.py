"""
80B 모델을 사용하여 32B 모델의 CoT를 검토하는 스크립트

사용법:
    python -m api_inference.scripts.verify_with_80b \
        --input data/train_cot_logit_both_fail.csv \
        --output data/train_80b_verification.csv
"""
import asyncio
import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

# 80B 모델 설정
VERIFIER_80B_CONFIG = {
    "base_url": "http://10.28.228.79:30390/v1",
    "api_key": "EMPTY",
    "model_name": "local_model",
    "timeout": 180,  # 80B는 더 오래 걸릴 수 있음
    "temperature": 0.0,
    "max_tokens": 2048,  # 더 긴 응답 가능
}

# 검토용 프롬프트
VERIFIER_SYSTEM_PROMPT = """당신은 수능/공무원 시험 문제를 검토하는 전문가입니다.
다른 모델의 추론 과정을 검토하고, 오류가 있다면 지적하여 올바른 답을 제시해주세요.

반드시 마지막 줄에 "정답: (숫자)" 형식으로 답을 출력하세요."""

VERIFIER_USER_PROMPT = """[지문]
{paragraph}

[질문]
{question}
{question_plus}

[선택지]
{choices}

[32B 모델의 추론]
{previous_cot}

위 추론에 대해 검토해주세요:
1. 논리적 오류나 사실 오류가 있는지 확인하세요.
2. 누락된 정보나 잘못된 판단이 있는지 확인하세요.
3. 각 선택지에 대한 분석이 올바른지 검토하세요.
4. 최종적으로 올바른 답을 제시하세요.

검토 결과를 상세히 서술하고, 마지막 줄에 "정답: (숫자)" 형식으로 답하세요."""


def parse_problems(problems_str: str) -> dict:
    """
    problems 컬럼을 파싱합니다.
    예: "{'question': '...', 'choices': [...], 'answer': 1}"
    """
    if pd.isna(problems_str) or not problems_str:
        return {}
    
    try:
        # ast.literal_eval을 사용하는 게 더 안전함
        import ast
        return ast.literal_eval(problems_str)
    except (ValueError, SyntaxError):
        try:
            # JSON 형식으로 시도
            problems_str = str(problems_str).replace("'", '"')
            return json.loads(problems_str)
        except:
            # 파싱 실패 시 빈 딕셔너리 반환
            print(f"Warning: Failed to parse problems: {problems_str[:100]}...")
            return {}


def parse_answer_from_response(response: str, num_choices: int = 5) -> int:
    """응답에서 정답 숫자를 추출합니다."""
    import re
    
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
    
    return 0


async def verify_single_row(
    row: pd.Series,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    단일 행에 대해 80B 모델로 검증합니다.
    """
    async with semaphore:
        # problems 파싱
        problems = parse_problems(row.get('problems', ''))
        question = problems.get('question', '')
        choices_list = problems.get('choices', [])
        
        if not question or not choices_list:
            return {
                "id": row['id'],
                "80b_answer": 0,
                "80b_cot": "Error: Failed to parse problems column",
                "80b_raw_response": "Error: Failed to parse problems column",
            }
        
        num_choices = len(choices_list)
        
        # 선택지 텍스트 구성
        choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices_list)])
        
        # question_plus 처리
        question_plus = row.get('question_plus', '')
        if pd.isna(question_plus) or str(question_plus).strip() in ['', 'nan', 'None']:
            question_plus_text = ""
        else:
            question_plus_text = f"\n<보기>\n{question_plus}"
        
        # 프롬프트 구성
        user_message = VERIFIER_USER_PROMPT.format(
            paragraph=row['paragraph'],
            question=question,
            question_plus=question_plus_text,
            choices=choices_text,
            previous_cot=row['raw_response']
        )
        
        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # 재시도 로직 (최대 3회)
        max_retries = 3
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                # 80B 모델 호출
                response = await client.chat.completions.create(
                    model=VERIFIER_80B_CONFIG["model_name"],
                    messages=messages,
                    temperature=VERIFIER_80B_CONFIG["temperature"],
                    max_tokens=VERIFIER_80B_CONFIG["max_tokens"],
                )
                
                content = response.choices[0].message.content
                answer = parse_answer_from_response(content, num_choices)
                
                return {
                    "id": row['id'],
                    "80b_answer": answer,
                    "80b_cot": content,
                    "80b_raw_response": content,
                }
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                error_msg = str(e)
                
                # 마지막 시도가 아니면 재시도
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # 지수 백오프
                    continue
                else:
                    # 모든 재시도 실패
                    return {
                        "id": row['id'],
                        "80b_answer": 0,
                        "80b_cot": f"Error ({error_type}): {error_msg}",
                        "80b_raw_response": f"Error ({error_type}): {error_msg}",
                    }


async def main():
    parser = argparse.ArgumentParser(
        description="80B 모델로 32B 모델의 CoT를 검증"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 CSV 파일 경로 (train_cot_logit_both_fail.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 CSV 파일 경로"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="동시 요청 수 (기본값: 3, 80B는 느리므로 낮게 설정)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="테스트용 샘플 개수 제한"
    )
    
    args = parser.parse_args()
    
    # CSV 읽기
    print(f"Loading CSV: {args.input}")
    df = pd.read_csv(args.input)
    
    if args.sample:
        df = df.head(args.sample)
        print(f"Using sample: {args.sample} rows")
    
    print(f"Total rows: {len(df)}")
    
    # 80B 모델 클라이언트 초기화
    print(f"\n80B Model Configuration:")
    print(f"  Base URL: {VERIFIER_80B_CONFIG['base_url']}")
    print(f"  Model: {VERIFIER_80B_CONFIG['model_name']}")
    print(f"  Timeout: {VERIFIER_80B_CONFIG['timeout']}s")
    
    client = AsyncOpenAI(
        base_url=VERIFIER_80B_CONFIG["base_url"],
        api_key=VERIFIER_80B_CONFIG["api_key"],
        timeout=VERIFIER_80B_CONFIG["timeout"],
        max_retries=0,  # 수동 재시도 로직 사용
    )
    
    # 연결 테스트
    print("\nTesting connection to 80B model...")
    try:
        test_response = await client.chat.completions.create(
            model=VERIFIER_80B_CONFIG["model_name"],
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10,
        )
        print("✅ Connection successful!")
    except Exception as e:
        print(f"❌ Connection failed: {type(e).__name__}: {str(e)}")
        print("\nPossible issues:")
        print("  1. 80B model server is not running")
        print("  2. Wrong API URL or port")
        print("  3. Network connectivity issue")
        print("\nContinuing anyway... (errors will be logged)")
    
    # 세마포어 설정
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    # 각 행에 대해 검증
    print(f"Starting verification with 80B model (max_concurrent={args.max_concurrent})...")
    tasks = [
        verify_single_row(row, client, semaphore)
        for _, row in df.iterrows()
    ]
    
    results = await tqdm_asyncio.gather(*tasks, desc="80B Verification")
    
    # 결과를 딕셔너리로 변환
    results_dict = {r['id']: r for r in results}
    
    # 원본 데이터프레임에 결과 추가
    df['80b_answer'] = df['id'].map(lambda x: results_dict.get(x, {}).get('80b_answer', 0))
    df['80b_cot'] = df['id'].map(lambda x: results_dict.get(x, {}).get('80b_cot', ''))
    df['80b_raw_response'] = df['id'].map(lambda x: results_dict.get(x, {}).get('80b_raw_response', ''))
    
    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nResults saved to: {output_path}")
    
    # 통계 출력
    total = len(df)
    valid_answers = (df['80b_answer'] > 0).sum()
    print(f"\n=== Statistics ===")
    print(f"Total problems: {total}")
    print(f"Valid answers (1-5): {valid_answers} ({valid_answers/total*100:.1f}%)")
    print(f"Failed to parse: {total - valid_answers} ({(total-valid_answers)/total*100:.1f}%)")
    
    # 32B vs 80B 비교
    if 'answer' in df.columns:
        correct_32b = (df['llm_pred'] == df['answer']).sum()
        correct_80b = (df['80b_answer'] == df['answer']).sum()
        print(f"\n=== Accuracy ===")
        print(f"32B (llm_pred): {correct_32b}/{total} ({correct_32b/total*100:.1f}%)")
        print(f"80B (80b_answer): {correct_80b}/{total} ({correct_80b/total*100:.1f}%)")
        print(f"Improvement: {correct_80b - correct_32b} problems")


if __name__ == "__main__":
    asyncio.run(main())

