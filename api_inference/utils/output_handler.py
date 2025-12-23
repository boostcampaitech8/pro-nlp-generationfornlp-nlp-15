"""
결과 저장 모듈
- output_시간.csv 파일명 생성
- output_raw_시간.csv 원문 텍스트 포함 파일 생성
- DataFrame 저장 로직
"""
import os
import time
import pandas as pd
from typing import List, Dict, Tuple


def generate_output_filename(output_dir: str, prefix: str = "output") -> str:
    """
    고유한 출력 파일명 생성
    파일명 형식: {prefix}_XXXXXXX.csv (시간 기반 7자리 숫자)
    
    Args:
        output_dir: 출력 디렉토리 경로
        prefix: 파일명 접두사 (기본값: "output")
    
    Returns:
        전체 파일 경로
    """
    # 현재 시간(초)의 뒤 7자리 사용
    time_suffix = int(time.time()) % (10 ** 7)
    filename = f"{prefix}_{time_suffix}.csv"
    return os.path.join(output_dir, filename)


def save_results(
    results: List[Dict[str, str]],
    output_dir: str,
    filename: str = None
) -> str:
    """
    추론 결과를 CSV 파일로 저장 (id, answer만 포함)
    
    Args:
        results: 결과 리스트 [{'id': ..., 'answer': ...}, ...]
        output_dir: 출력 디렉토리
        filename: 파일명 (None이면 자동 생성)
    
    Returns:
        저장된 파일 경로
    """
    if filename is None:
        output_path = generate_output_filename(output_dir, "output")
    else:
        output_path = os.path.join(output_dir, filename)
    
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # DataFrame 생성 및 저장 (id, answer만)
    df = pd.DataFrame(results)[['id', 'answer']]
    df.to_csv(output_path, index=False)
    
    return output_path


def save_results_with_raw(
    results: List[Dict[str, str]],
    output_dir: str,
    type_stats: Dict[str, int] = None
) -> Tuple[str, str]:
    """
    추론 결과를 두 개의 CSV 파일로 저장
    - output_시간.csv: id, answer만 포함
    - output_raw_시간.csv: id, answer, question_type, raw_response 포함
    - output_type_시간.csv: 문제 유형별 통계 (type_stats가 제공된 경우)
    
    Args:
        results: 결과 리스트 [{'id': ..., 'answer': ..., 'raw_response': ..., 'question_type': ...}, ...]
        output_dir: 출력 디렉토리
        type_stats: 문제 유형별 통계 딕셔너리 (optional)
    
    Returns:
        (output 파일 경로, raw output 파일 경로) 튜플
    """
    # 시간 suffix 생성 (모든 파일에 동일하게 적용)
    time_suffix = int(time.time()) % (10 ** 7)
    
    output_path = os.path.join(output_dir, f"output_{time_suffix}.csv")
    raw_output_path = os.path.join(output_dir, f"output_raw_{time_suffix}.csv")
    
    # 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # output.csv 저장 (id, answer만)
    df_output = pd.DataFrame(results)[['id', 'answer']]
    df_output.to_csv(output_path, index=False)
    
    # output_raw.csv 저장 (id, answer, question_type, raw_response)
    df_raw = pd.DataFrame(results)
    
    # 필요한 컬럼만 선택
    available_columns = ['id', 'answer', 'question_type', 'raw_response']
    df_raw = df_raw[[col for col in available_columns if col in df_raw.columns]]
    
    df_raw.to_csv(raw_output_path, index=False, encoding='utf-8-sig')
    
    # output_type.csv 저장 (문제 유형별 통계)
    if type_stats is not None:
        type_output_path = os.path.join(output_dir, f"output_type_{time_suffix}.csv")
        df_type = pd.DataFrame([
            {'question_type': qtype, 'count': count}
            for qtype, count in type_stats.items()
        ])
        df_type.to_csv(type_output_path, index=False, encoding='utf-8-sig')
    
    return output_path, raw_output_path

