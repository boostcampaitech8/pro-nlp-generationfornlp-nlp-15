# Multi-Agent 구현 방식 비교 분석

## 📋 개요

두 파일은 서로 다른 Multi-Agent 아키텍처를 사용하여 문제 풀이 성능을 향상시키려 합니다.

---

## 🏗️ 방식 1: `api_inference.py` - 클래스 기반 Multi-Agent

### 구조도

```
┌─────────────────────────────────────────────────────────────┐
│                    api_inference.py                         │
│                  (use_multi_agent=True)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   MultiAgentProcessor            │
        │   (조율자 역할)                   │
        └────────────┬─────────────────────┘
                     │
                     │ 1. Primary Agent 실행
                     ▼
        ┌──────────────────────────────────┐
        │   _run_primary_agent()           │
        │   - CoT 방식으로 문제 풀이       │
        │   - 동일 모델 (AsyncAPIClient)   │
        │   - MAX_RETRY=3                  │
        └────────────┬─────────────────────┘
                     │
                     │ Primary Answer 파싱
                     │ (answer < 1 or > 5?)
                     ▼
        ┌──────────────────────────────────┐
        │   needs_verification?            │
        │   YES ───────────────────┐       │
        │   NO ────────────► 최종 답       │
        └──────────────────────────┘       │
                                           │
                                           ▼
        ┌──────────────────────────────────────────────────┐
        │           VerifierAgent                          │
        │   (동일 모델, 두 가지 방법 병렬 실행)            │
        └──────┬───────────────────────────────┬───────────┘
               │                               │
               ▼                               ▼
    ┌──────────────────────┐      ┌──────────────────────┐
    │   _re_solve()        │      │   _review_cot()      │
    │   문제 재풀이        │      │   CoT 검토           │
    │                      │      │                      │
    │   독립적으로 다시 풀기│      │   Primary CoT 검토   │
    │   (비교적 간단한      │      │   오류 지적 후 수정  │
    │    프롬프트)          │      │   (더 복잡한        │
    │                      │      │    프롬프트)         │
    └──────────┬───────────┘      └──────────┬───────────┘
               │                              │
               └──────────┬───────────────────┘
                          │ asyncio.gather() 병렬 실행
                          ▼
        ┌──────────────────────────────────┐
        │   _decide_final_answer()         │
        │   - both_agree: 둘 다 일치       │
        │   - vote_re_solve: 재풀이 우선    │
        │   - re_solve: 재풀이만 유효       │
        │   - review_cot: 검토만 유효       │
        │   - both_failed: 둘 다 실패       │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │   MultiAgentResult               │
        │   - final_answer                 │
        │   - primary_answer               │
        │   - verifier_used                │
        │   - verifier_method              │
        │   - verifier_answer              │
        └──────────────────────────────────┘
```

### 특징

1. **클래스 기반 설계**
   - `MultiAgentProcessor`: 전체 흐름 조율
   - `VerifierAgent`: 검증 전용 클래스
   - `MultiAgentResult`: 구조화된 결과 객체 (dataclass)

2. **Verifier 전략: 이중 검증**
   - **방법 1**: `_re_solve()` - 문제를 처음부터 다시 풀기 (독립적)
   - **방법 2**: `_review_cot()` - Primary Agent의 CoT를 검토하여 수정
   - 두 방법을 **병렬로 실행** (`asyncio.gather()`)
   - 최종 답 결정 로직으로 통합

3. **모델 사용**
   - Primary Agent와 Verifier 모두 **동일 모델** (설정에서 지정)
   - `AsyncAPIClient`를 공유하여 사용

4. **Verifier 호출 조건**
   - Primary Agent의 답이 유효하지 않을 때만 (answer < 1 or > 5)
   - 파싱 실패 시에만 호출

5. **결과 구조**
   ```python
   MultiAgentResult(
       id, final_answer, answer,
       primary_answer, primary_raw_response,
       verifier_used, verifier_method,  # "both_agree", "re_solve", etc.
       verifier_answer, verifier_raw_response,
       question_type
   )
   ```

### 코드 흐름

```python
# 1. MultiAgentProcessor 초기화
processor = MultiAgentProcessor(client, system_prompt, ...)

# 2. 문제 처리
result = await processor.process_single_item(item, semaphore)

# 3. 내부 흐름
#    a) _run_primary_agent() → PrimaryAgentResult
#    b) needs_verification 체크
#    c) VerifierAgent.verify() 호출
#       - _re_solve() + _review_cot() 병렬 실행
#       - _decide_final_answer()로 최종 결정
#    d) MultiAgentResult 반환
```

---

## 🏗️ 방식 2: `api_inference_agent.py` - 함수 기반 80B Verifier

### 구조도

```
┌─────────────────────────────────────────────────────────────┐
│              api_inference_agent.py                         │
│          (80B Verifier 전용 Multi-Agent)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   process_single_item_with_80b() │
        │   (함수 기반, 직접 처리)          │
        └────────────┬─────────────────────┘
                     │
                     │ 1. Primary Agent (32B) 실행
                     ▼
        ┌──────────────────────────────────┐
        │   Primary Client                 │
        │   (AsyncAPIClient - 32B 모델)    │
        │   - CoT 방식으로 문제 풀이       │
        │   - MAX_RETRY=5                  │
        └────────────┬─────────────────────┘
                     │
                     │ Primary Answer 파싱
                     │ primary_response 저장
                     ▼
        ┌──────────────────────────────────┐
        │   needs_verification?            │
        │   - verify_all=True: 모든 문제    │
        │   - verify_all=False: 파싱 실패만 │
        │   - verify_threshold: 확신도 기준 │
        │                                  │
        │   YES ──────────────┐            │
        │   NO ────► 최종 답   │            │
        └──────────────────────┘            │
                                           │
                                           ▼
        ┌──────────────────────────────────┐
        │      Verifier80BAgent            │
        │   (80B 모델 전용, 단일 방법)      │
        │                                  │
        │   base_url: 80B 서버             │
        │   model_name: 80B 모델           │
        │   timeout: 180초                 │
        │   max_tokens: 2048               │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │   verify()                       │
        │   단일 검증 방법                  │
        │                                  │
        │   [System Prompt]                │
        │   - 간결한 응답 유도             │
        │   - 반복 방지                    │
        │                                  │
        │   [User Prompt]                  │
        │   - 원본 문제                    │
        │   - Primary CoT 포함             │
        │   - "검토해주세요" 요청           │
        │   - "정답: (숫자)" 형식 요구     │
        └────────────┬─────────────────────┘
                     │
                     │ 80B 모델 호출 (max_retries=3)
                     ▼
        ┌──────────────────────────────────┐
        │   parse_answer_from_response()   │
        │   - "정답: X" 패턴 추출          │
        │   - 응답 끝부분 숫자 추출         │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │   최종 답 결정                    │
        │   - verifier_answer > 0?         │
        │     → verifier_answer 사용       │
        │   - else: primary_answer 사용    │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │   결과 딕셔너리                   │
        │   {                               │
        │     "id", "answer",              │
        │     "raw_response":               │
        │       "[Primary Agent]..."       │
        │       "[80B Verifier]...",       │
        │     "multi_agent_info": {...}    │
        │   }                               │
        └──────────────────────────────────┘
```

### 특징

1. **함수 기반 설계**
   - `process_single_item_with_80b()`: 함수로 직접 처리
   - `Verifier80BAgent`: 80B 전용 클래스 (간단한 구조)
   - 결과는 딕셔너리로 반환 (dataclass 아님)

2. **Verifier 전략: 단일 검증 (CoT 검토)**
   - **방법**: Primary Agent의 CoT를 80B 모델이 검토
   - Primary CoT를 프롬프트에 포함하여 검토 요청
   - 독립적인 재풀이 없음

3. **모델 사용**
   - **Primary Agent**: 32B 모델 (`AsyncAPIClient`)
   - **Verifier**: 80B 모델 (`AsyncOpenAI`, 별도 서버)
   - 서로 다른 모델 사용 (더 큰 모델로 검증)

4. **Verifier 호출 조건**
   - `verify_all=True`: 모든 문제 검증
   - `verify_all=False`: 파싱 실패 시만
   - `verify_threshold`: 확신도 기반 (향후 logit 기반 구현 예정)

5. **특별 기능**
   - **CSV 모드**: 기존 32B CoT 결과 CSV를 읽어 80B Verifier만 실행
   - **System Prompt 최적화**: 간결한 응답, 반복 방지

6. **결과 구조**
   ```python
   {
       "id": str,
       "answer": str,
       "raw_response": str,  # "[Primary Agent]...\n[80B Verifier]..."
       "question_type": str,
       "multi_agent_info": {
           "primary_answer": int,
           "verifier_used": bool,
           "verifier_answer": int | None
       }
   }
   ```

### 코드 흐름

```python
# 1. Verifier80BAgent 초기화
verifier_80b = Verifier80BAgent(config)

# 2. 문제 처리
result = await process_single_item_with_80b(
    primary_client, verifier_80b, item, ...
)

# 3. 내부 흐름
#    a) Primary Agent (32B) 실행 → primary_response
#    b) needs_verification 체크 (verify_all / 파싱 실패)
#    c) Verifier80BAgent.verify() 호출
#       - 80B 모델로 Primary CoT 검토
#       - 단일 응답에서 답 추출
#    d) 최종 답 결정 (verifier > primary)
#    e) 딕셔너리 반환
```

---

## 🔍 핵심 차이점 비교표

| 항목 | `api_inference.py` | `api_inference_agent.py` |
|------|-------------------|-------------------------|
| **설계 방식** | 클래스 기반 (Processor 패턴) | 함수 기반 (직접 처리) |
| **Primary Agent** | 동일 모델 (설정에서 지정) | 32B 모델 (고정) |
| **Verifier Agent** | 동일 모델 (설정에서 지정) | 80B 모델 (고정) |
| **Verifier 전략** | 이중 검증 (재풀이 + CoT 검토) | 단일 검증 (CoT 검토만) |
| **Verifier 실행** | 병렬 (`asyncio.gather()`) | 순차 (단일 호출) |
| **Verifier 호출 조건** | 파싱 실패만 | `verify_all` 옵션 + 파싱 실패 |
| **결과 구조** | `MultiAgentResult` (dataclass) | 딕셔너리 |
| **최종 답 결정** | 복잡한 로직 (5가지 케이스) | 단순 (verifier 우선) |
| **CSV 모드** | ❌ | ✅ (80B만 실행) |
| **재시도 횟수** | Primary: 3, Verifier: 1 | Primary: 5, Verifier: 3 |
| **확장성** | 높음 (모델 변경 쉬움) | 낮음 (80B 전용) |

---

## 🎯 각 방식의 장단점

### 방식 1: `api_inference.py` (클래스 기반)

**장점:**
- ✅ 이중 검증으로 더 높은 정확도 기대 가능
- ✅ 유연한 설계 (모델 변경 쉬움)
- ✅ 구조화된 결과 객체 (타입 안정성)
- ✅ Verifier 전략이 다양 (재풀이 vs 검토)

**단점:**
- ❌ Verifier 호출 시 API 호출이 2배 (재풀이 + 검토)
- ❌ 더 복잡한 코드 구조
- ❌ 최종 답 결정 로직이 복잡함

### 방식 2: `api_inference_agent.py` (80B Verifier)

**장점:**
- ✅ 더 큰 모델(80B)로 검증하여 성능 향상 가능
- ✅ 단일 검증으로 API 호출 최소화
- ✅ 간단한 코드 구조
- ✅ CSV 모드로 재검증 용이
- ✅ `verify_all` 옵션으로 모든 문제 검증 가능

**단점:**
- ❌ 80B 모델이 느림 (추론 시간 증가)
- ❌ 단일 검증 방법 (유연성 낮음)
- ❌ 80B 전용 (다른 모델로 변경 어려움)

---

## 📊 실제 사용 시나리오

### 시나리오 1: 빠른 실험, 모델 비교
→ **`api_inference.py`** 사용
- 동일 모델로 Primary + Verifier 구성
- 다양한 모델 조합 테스트 가능

### 시나리오 2: 최고 성능 추구
→ **`api_inference_agent.py`** 사용
- 32B로 빠르게 풀이, 80B로 정확하게 검증
- `verify_all=True`로 모든 문제 검증

### 시나리오 3: 기존 결과 재검증
→ **`api_inference_agent.py`** CSV 모드
- 32B CoT 결과 CSV를 읽어 80B로만 재검증
- 시간/비용 절약

---

## 🔧 코드 레벨 비교

### Verifier 호출 방식

**방식 1:**
```python
# 두 방법을 병렬로 실행
re_solve_task = self._re_solve(...)
review_task = self._review_cot(...)
(re_solve_answer, re_solve_response), (review_answer, review_response) = await asyncio.gather(
    re_solve_task, review_task
)
# 최종 답 결정 (복잡한 로직)
final_answer, method_used, confidence = self._decide_final_answer(...)
```

**방식 2:**
```python
# 단일 검증만 실행
verifier_answer, verifier_response = await verifier_80b.verify(
    paragraph=item['paragraph'],
    question=item['question'],
    primary_cot=primary_response,  # Primary CoT 포함
    ...
)
# 최종 답 결정 (단순)
if verifier_used and verifier_answer > 0:
    final_answer = verifier_answer
else:
    final_answer = primary_answer
```

### 프롬프트 구조

**방식 1 - 재풀이:**
```
[문제 정보]
위 문제를 풀고, 마지막 줄에 "정답: (숫자)" 형식으로 답을 출력하세요.
```

**방식 1 - CoT 검토:**
```
[문제 정보]
[이전 풀이]
{previous_solution}

위 풀이를 검토해주세요:
1. 각 선택지 판단이 올바른지 확인하세요.
2. 논리적 오류가 있다면 지적해주세요.
3. 최종 답이 올바른지 판단하세요.
```

**방식 2 - 80B 검토:**
```
[지문]
{paragraph}
[질문]
{question}
[선택지]
{choices}
[1차 모델의 추론]
{primary_cot}

위 추론에 대해 검토해주세요:
1. 논리적 오류나 사실 오류가 있는지 확인하세요.
2. 누락된 정보나 잘못된 판단이 있는지 확인하세요.
3. 각 선택지에 대한 분석이 올바른지 검토하세요.
4. 최종적으로 올바른 답을 제시하세요.

검토 결과를 간결하고 핵심적으로 서술하고, 반드시 마지막 줄에 "정답: (숫자)" 형식으로 답하세요.
```

---

## 🎓 결론

두 방식은 서로 다른 철학을 가지고 있습니다:

- **`api_inference.py`**: **"정확도 향상을 위한 다각도 검증"**
  - 같은 모델의 다양한 접근 방식 활용
  - 이중 검증으로 신뢰성 향상

- **`api_inference_agent.py`**: **"큰 모델로 확실하게 검증"**
  - 작은 모델(32B)로 빠르게, 큰 모델(80B)로 확실하게
  - 실용적이고 효율적인 접근

사용자는 목적에 맞는 방식을 선택하면 됩니다! 🚀

