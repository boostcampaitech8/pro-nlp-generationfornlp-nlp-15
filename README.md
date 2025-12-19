# baselined refactored v1

멀티 패키지 구조로 **실험(실행 코드)** 와 **공통 모듈(재사용 코드)** 를 분리해 둔 프로젝트입니다.

## 구조

- `baseline/` : 실행 가능한 “실험 패키지”
  - `main.py` 학습 엔트리포인트
  - `inference.py` 추론 엔트리포인트(있는 경우)
  - `configs/` pydantic schema + yaml 로더 + `config.yaml`
  - `models/` 모델/토크나이저 로더
  - `trainer/` SFT runner, metrics

- `common/` : 여러 실험에서 공유하는 “라이브러리”
  - `data/` CSV → examples → messages → HF Dataset/tokenize
  - `prompts/` system prompt / formatter / templates
  - `tokenization/` chat_template 등
  - `utils/` logger, wandb helper

- `outputs/` : 실험 결과 저장(output_dir 예시)

## 실행 (uv, 루트에서)

파일 직접 실행하지 말고, 항상 패키지 모듈로 실행합니다.

학습:
```bash
uv run python -m baseline.train --config baseline/configs/config.yaml
```
추론(있는 경우):
```bash
uv run python -m baseline.inference --config baseline/configs/config.yaml
```

설정(config)
	•	코드 스키마: baseline/configs/schema.py
	•	로더: baseline/configs/load.py
	•	실행 설정: baseline/configs/config.yaml
