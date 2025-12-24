"""
결과 저장 모듈
- output_시간.csv 파일명 생성
- output_raw_시간.csv 원문 텍스트 포함 파일 생성
- DataFrame 저장 로직
- 실시간/배치 저장 지원
"""
import os
import time
import threading
import pandas as pd
from typing import List, Dict, Tuple, Optional


class StreamingResultSaver:
    """실시간/배치 결과 저장 클래스"""
    
    def __init__(self, output_dir: str, batch_size: int = 50):
        """
        Args:
            output_dir: 출력 디렉토리
            batch_size: 배치 저장 크기 (기본값: 50)
        """
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.results: List[Dict] = []
        self.lock = threading.Lock()
        
        # 시간 suffix 생성 (세션 내 모든 파일에 동일하게 적용)
        self.time_suffix = int(time.time()) % (10 ** 7)
        
        # 파일 경로 설정
        self.output_path = os.path.join(output_dir, f"output_{self.time_suffix}.csv")
        self.raw_output_path = os.path.join(output_dir, f"output_raw_{self.time_suffix}.csv")
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 헤더 작성 (파일 초기화)
        self._init_files()
    
    def _init_files(self):
        """CSV 파일 헤더 초기화"""
        # output.csv 헤더
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write('id,answer\n')
        
        # output_raw.csv 헤더
        with open(self.raw_output_path, 'w', encoding='utf-8-sig') as f:
            f.write('id,answer,question_type,raw_response\n')
    
    def add_result(self, result: Dict) -> None:
        """
        결과 추가 및 배치 저장
        
        Args:
            result: 단일 결과 딕셔너리
        """
        with self.lock:
            self.results.append(result)
            
            # 배치 크기에 도달하면 저장
            if len(self.results) >= self.batch_size:
                self._flush()
    
    def _flush(self) -> None:
        """현재까지의 결과를 파일에 저장 (append 모드)"""
        if not self.results:
            return
        
        # output.csv에 append
        with open(self.output_path, 'a', encoding='utf-8') as f:
            for r in self.results:
                f.write(f"{r['id']},{r['answer']}\n")
        
        # output_raw.csv에 append (raw_response의 쉼표/줄바꿈 처리)
        with open(self.raw_output_path, 'a', encoding='utf-8-sig') as f:
            for r in self.results:
                # CSV 이스케이프 처리
                raw_response = r.get('raw_response', '').replace('"', '""')
                question_type = r.get('question_type', '')
                f.write(f'{r["id"]},{r["answer"]},{question_type},"{raw_response}"\n')
        
        self.results.clear()
    
    def finalize(self, type_stats: Optional[Dict[str, int]] = None) -> Tuple[str, str]:
        """
        남은 결과 저장 및 최종화
        
        Args:
            type_stats: 문제 유형별 통계 (optional)
        
        Returns:
            (output 파일 경로, raw output 파일 경로)
        """
        with self.lock:
            self._flush()
        
        # output_type.csv 저장
        if type_stats is not None:
            type_output_path = os.path.join(self.output_dir, f"output_type_{self.time_suffix}.csv")
            df_type = pd.DataFrame([
                {'question_type': qtype, 'count': count}
                for qtype, count in type_stats.items()
            ])
            df_type.to_csv(type_output_path, index=False, encoding='utf-8-sig')
        
        return self.output_path, self.raw_output_path


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

