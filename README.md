# 수능형 문제 풀이 모델 생성 Project

---

## 프로젝트 폴더 구조

```text
pro-nlp-generationfornlp-15/
├── common/                                # 팀 내 공통으로 사용하는 data, prompts, utils 관리
│
├── api_inference/                         # vLLM 서버에서 LLM을 서빙받아 추론
│   ├── agents/                            # Multi-Agent 시스템 구성 요소
|   |   ├── primary_agent.py               # 1차 Agent: 32B 모델로 CoT 방식 문제 풀이
|   |   ├── verifier_80b_agent.py          # 2차 Verifier: 80B 모델로 답안 검증 및 재풀이
|   |   └── multi_agent_processor_80b.py   # 전체 파이프라인 조율 (Primary → Verifier)
|   |
│   ├── config/                            # 설정 파일
│   |   └── inference_agent.yaml           # Multi-Agent 추론 설정 (API 엔드포인트, 프롬프트, 검증 조건 등)
|   |
│   ├── prompts/                           # 프롬프트 템플릿
|   │   ├── templates.py                   # 문제 유형별 프롬프트 템플릿 및 시스템 프롬프트
|   │   └── question_type.py               # 문제 유형 분류 관련 상수
|   |
│   ├── utils/                             # 유틸리티 모듈
|   |   ├── answer_parser.py               # LLM 응답에서 정답(1-5) 추출
|   |   ├── api_client.py                  # vLLM 서버와의 비동기 HTTP 통신 클라이언트
|   |   ├── data_loader.py                 # CSV 데이터 로드 및 문제 유형 분류
|   |   ├── metrics.py                     # F1 Score, Accuracy 등 평가 지표 계산
|   |   └── output_handler.py              # 결과 저장 및 포맷팅
|   |
│   └── api_inference_agent.py             # Multi-Agent 추론 실행 스크립트 (메인 진입점)
│
├── data/
│   ├── train.csv                # 학습용 데이터 (id, paragraph, problems)
│   └── test.csv                 # 추론용 데이터 (id, paragraph, problems)
│
└── pyproject.toml                # 가상 환경 설정
````

---

## 파일 및 폴더 상세 설명

### Multi-Agent 추론 시스템 (`api_inference/`)

#### `api_inference/api_inference_agent.py`

* **Multi-Agent 추론 시스템**
* 1차 Agent(default: 32B)로 CoT 방식 문제 풀이 수행
* 2차 Verifier(default: 80B)로 답안 검증 및 재풀이
* YAML 설정 파일 기반 동작
* 비동기 병렬 처리로 성능 최적화
* Train 모드: F1 Score 계산 및 WandB 기록
* Test 모드: 결과만 저장
* CSV 모드: 기존 32B CoT 결과 CSV 파일을 읽어와, 80B Verifier만 실행 가능


### api_inference/agents
- `agents/primary_agent.py`
  * **1차 Agent**: 32B 모델로 Chain of Thought 방식 문제 풀이
  * 문제 유형별 프롬프트 지원 (optional)
  * CoT 프롬프트로 단계별 추론 과정 생성
  * 재시도 로직 및 에러 핸들링 포함

- `agents/verifier_80b_agent.py`
  * **2차 Verifier**: 80B 모델로 Primary Agent의 답안 검증
  * Primary Agent의 응답과 문제를 함께 제공하여 재검토
  * 답안 파싱 실패 시 재풀이 수행

- `agents/multi_agent_processor_80b.py`
  * **파이프라인 조율**: Primary Agent와 80B Verifier의 흐름 관리
  * `verify_all` 옵션: 모든 문제 검증 또는 조건부 검증
  * 각 문제에 대한 최종 결과 통합

### api_inference/utils
- `utils/api_client.py`
  * vLLM 서버와의 비동기 HTTP 통신 처리
  * OpenAI API 호환 인터페이스 제공
  * 타임아웃, 재시도, 동시성 제어 지원

- `utils/data_loader.py`
  * CSV 데이터 로드 및 전처리
  * 문제 유형 분류 (LLM 기반, optional)
  * 샘플 크기 제한 및 데이터 필터링

- `utils/answer_parser.py`
  * LLM 응답에서 정답(1-5) 추출
  * 다양한 응답 포맷 지원 (`정답: 3`, `답: 2`, `3번` 등)
  * 파싱 실패 시 0 반환 (Verifier 트리거)

- `utils/metrics.py`
  * F1 Score, Accuracy 계산
  * 문제 유형별 성능 분석
  * 평가 리포트 생성

- `utils/output_handler.py`
  * 결과를 CSV 형식으로 저장
  * Raw 응답 저장 (디버깅 및 분석용)
  * 실시간 스트리밍 저장 지원 (optional)

### api_inference/config
- `config/inference_agent.yaml`
  * **설정 파일**: 모든 추론 파라미터 관리
  * API 엔드포인트 및 모델 설정
  * 프롬프트, 검증 조건, 동시성 제어 등
  * 코드 수정 없이 설정만으로 실험 가능

---

## 실행 방법

### 1. 기본 실행 (전체 파이프라인)

```bash
python -m api_inference.api_inference_agent --config api_inference/config/inference_agent.yaml
```

### 2. 설정 파일 수정

`api_inference/config/inference_agent.yaml` 파일에서 다음 설정 가능:

* **API 엔드포인트**: Primary Agent 및 80B Verifier URL
* **검증 조건**: `verify_all: true/false` (모든 문제 검증 또는  파싱 실패된 답변 한정 실행)
* **프롬프트**: `use_cot, use_type_specific_prompt : true/false` (CoT 사용 여부, 유형별 프롬프트)
* **동시성 제어**: `max_concurrent` (병렬 처리 수)
* **모드**: `mode: "train"/"test"`

### 3. 출력 결과

* **Train 모드**:
  * `data/{output_dir}/output_*.csv`: 최종 결과 (id, answer)
  * `data/{output_dir}/output_raw_*.csv`: Raw 응답 (CoT 답변 확인용)
  * 터미널에 F1 Score, 문제 유형별 통계 출력
  * WandB에 Metric 기록 (설정된 경우)

* **Test 모드**:
  * `data/{output_dir}/output_*.csv`: 최종 결과 (id, answer)
  * `data/{output_dir}/output_raw_*.csv`: Raw 응답 (CoT 답변 확인용)
