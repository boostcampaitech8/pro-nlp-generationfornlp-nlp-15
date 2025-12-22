# src/metrics.py
import torch
import numpy as np
import evaluate
from sklearn.metrics import f1_score


class CustomMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.acc_metric = evaluate.load("accuracy")
        self.int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

        # 정답 토큰("1"~"5")의 ID를 미리 찾아둡니다.
        # tokenizer.vocab["1"] 방식보다 안전한 convert_tokens_to_ids 사용
        self.logit_idx = self.tokenizer.convert_tokens_to_ids(["1", "2", "3", "4", "5"])

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        모델의 Logits 중 정답 토큰(1~5)에 해당하는 부분만 추출합니다.

        동적 인덱스 탐색:
        - labels에서 첫 번째 valid 토큰(-100이 아닌) 위치를 찾음
        - 그 직전 위치의 logits가 정답을 예측하는 logits임
        - 이 방식은 토크나이저나 EOS 토큰 구조에 관계없이 안정적으로 동작
        """
        logits = logits if not isinstance(logits, tuple) else logits[0]

        batch_size = logits.size(0)
        batch_logits = []

        # 디버그: 첫 번째 배치의 labels 구조 출력 (1회만)
        if not hasattr(self, "_debug_logged"):
            self._debug_logged = True
            print(f"\n[DEBUG] labels shape: {labels.shape}")
            print(f"[DEBUG] labels[0] (last 20): {labels[0][-20:].tolist()}")
            valid_count = (labels[0] != -100).sum().item()
            print(f"[DEBUG] valid labels count in first sample: {valid_count}")
            if valid_count > 0:
                valid_mask = labels[0] != -100
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                print(
                    f"[DEBUG] first valid pos: {valid_indices[0].item()}, last valid pos: {valid_indices[-1].item()}"
                )
                valid_labels = labels[0][valid_mask].tolist()
                print(
                    f"[DEBUG] valid labels decoded: {self.tokenizer.decode(valid_labels)}"
                )

        for i in range(batch_size):
            # labels에서 첫 번째 valid 토큰 위치 찾기 (labels != -100)
            valid_mask = labels[i] != -100
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            if len(valid_indices) > 0:
                # 첫 번째 valid label 위치의 직전 = 정답 예측 위치
                first_valid_pos = valid_indices[0].item()
                pred_pos = first_valid_pos - 1
                batch_logits.append(logits[i, pred_pos, self.logit_idx])
            else:
                # fallback: valid label이 없는 경우 (예외 상황)
                print(f"[DEBUG] WARNING: No valid labels in sample {i}")
                batch_logits.append(logits[i, -1, self.logit_idx])

        return torch.stack(batch_logits)

    def compute_metrics(self, evaluation_result):
        """
        User provided specific metric logic
        """
        logits, labels = evaluation_result

        # 1. 라벨 디코딩
        # -100(ignore_index)을 pad_token_id로 치환
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 2. 텍스트 파싱 (<end_of_turn> 제거 및 매핑)
        # Gemma 스타일: "1<end_of_turn>" -> "1"
        parsed_labels = []
        for label in decoded_labels:
            clean_label = label.split("<end_of_turn>")[0].strip()
            # 맵핑에 없는 값이 나올 경우(데이터 오류 등) 예외 처리 혹은 기본값 0
            parsed_labels.append(self.int_output_map.get(clean_label, 0))

        # 3. 예측값 추출 (Argmax)
        # logits는 이미 preprocess에서 (Batch, 5) 크기로 줄어들어 있음
        # dim=-1은 5개의 후보("1"~"5") 중 확률이 높은 인덱스(0~4)를 선택함
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        # 4. Macro F1-score 및 정확도 계산
        macro_f1 = f1_score(parsed_labels, predictions, average="macro")
        acc = self.acc_metric.compute(predictions=predictions, references=parsed_labels)

        return {
            "macro_f1": macro_f1,
            "accuracy": acc["accuracy"],
        }
