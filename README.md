# 수능형 문제 풀이 모델 생성 Project

---

## 📂 프로젝트 폴더 구조 (Project Structure)
koreasat_project/
├── configs/
│   └── train_config.yaml        # 모델, LoRA, 학습 하이퍼파라미터 통합 관리
│
├── data/
│   ├── train.csv                # 학습용 데이터 (id, paragraph, problems)
│   └── test.csv                 # 추론용 데이터 (id, paragraph, problems)
│
├── outputs/
│   └── final_adapter/           # 학습 완료 후 생성되는 최종 LoRA 가중치 및 설정
│
├── data_utils.py                # 데이터 로드, literal_eval 파싱, 동적 프롬프트 생성
├── model_utils.py               # 모델/토크나이저 로드, Chat Template, 프롬프트 상수
├── trainer.py                   # SFTTrainer 실행, Completion Only 학습 설정
├── metrics.py                   # Logit 비교 기반 정답 토큰 Accuracy 계산
├── main.py                      # 학습 실행 엔트리 포인트
├── inference.py                 # 추론 실행 및 output.csv 생성
└── requirements.txt             # 필수 라이브러리 목록


---

## 📝 파일 및 폴더 상세 설명

### 1. 주요 실행 파일

#### `main.py`
- 프로젝트의 **학습 시작점**
- `configs/train_config.yaml`을 로드하여 모델, 데이터, LoRA 설정 초기화
- `trainer.py`를 호출하여 학습 수행
- WandB 로깅 연동 가능

#### `inference.py`
- 학습 완료 후 실행
- `outputs/final_adapter`의 LoRA 가중치를 베이스 모델에 로드
- `data/test.csv`에 대해 추론 수행
- 최종 제출용 `output.csv` 생성

---

### 2. 핵심 유틸리티 모듈

#### `model_utils.py`
- 프로젝트 전반의 **프롬프트 및 모델 설정 중심**
- Gemma 전용 Chat Template 관리
- 공통 프롬프트 상수 정의
- `<보기>` 존재 여부에 따라 프롬프트를 동적으로 조립하는  
  `format_question_message` 함수 제공
- 모든 학습/추론 코드에서 동일한 포맷 사용을 보장

#### `data_utils.py`
- 데이터 전처리 전담 모듈
- CSV 내부의 문자열 딕셔너리를 `ast.literal_eval`로 안전하게 파싱
- HuggingFace `Dataset` 생성
- 토큰화 및 최대 길이(1024 토큰) 필터링 수행

#### `trainer.py`
- 실제 학습 로직 구현
- `trl.SFTTrainer` 기반 학습
- `DataCollatorForCompletionOnlyLM` 사용  
  → **질문 부분은 무시하고 모델 응답 부분만 학습**
- RTX 5080 기준 최적화 옵션 포함
  - bf16
  - tf32
  - Gradient Checkpointing

#### `metrics.py`
- 평가 로직 담당
- 단순 문자열 비교가 아닌,
  **1~5번 선택지 토큰의 Logits 점수 직접 비교**
- 가장 높은 확률을 가진 토큰을 정답으로 판별하여 Accuracy 계산

---

### 3. 설정 및 데이터 관리

#### `configs/`
- 모든 실험 설정을 YAML로 관리
- 코드 수정 없이 다음 항목 변경 가능:
  - Base Model
  - Batch Size
  - Learning Rate
  - Epoch
  - LoRA r / alpha / dropout
  - Gradient Accumulation

#### `data/`
- 원본 데이터 저장 디렉토리
- `literal_eval` 기반 파싱을 사용하므로  
  홑따옴표(`'`)가 포함된 파이썬 딕셔너리 형태도 처리 가능

---

## 실행 방법

### 1. 환경 구축
```bash
pip install -r requirements.txt

2. 학습 실행
python main.py

- WandB 설정 시 학습 로그 실시간 확인 가능

3. 추론 실행
python inference.py

4. 결과물
- 루트 디렉토리에 output.csv 생성
- 대회 제출용 파일로 바로 사용 가능
