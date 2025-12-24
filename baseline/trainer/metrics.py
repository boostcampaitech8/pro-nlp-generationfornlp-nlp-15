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
        self.logit_idx = self.tokenizer.convert_tokens_to_ids(["1", "2", "3", "4", "5"])

        # response_template 토큰 ID (정답 직전 위치 찾기용)
        # "<start_of_turn>model\n" 이후에 정답이 위치
        response_template = "<start_of_turn>model\n"
        self.response_template_ids = self.tokenizer.encode(
            response_template, add_special_tokens=False
        )

        self._debug_printed = False  # 디버깅 로그 1회만 출력

    def _find_response_start(self, input_ids) -> int:
        """
        input_ids(또는 labels)에서 정답 시작 위치를 찾습니다.

        completion_only_loss=True 환경에서는 labels의 prompt 부분이 -100으로
        마스킹되므로, -100이 아닌 첫 번째 토큰 위치를 정답 시작 위치로 반환합니다.

        Returns:
            정답 시작 위치 (0-indexed). 못 찾은 경우 -1.
        """
        if isinstance(input_ids, torch.Tensor):
            input_list = input_ids.tolist()
        else:
            input_list = list(input_ids)

        # 방법 1: -100이 아닌 첫 번째 토큰 위치 찾기 (completion_only_loss=True 환경)
        # labels가 마스킹된 경우: prompt 부분은 -100, completion 부분만 실제 토큰 ID
        for i, token_id in enumerate(input_list):
            if token_id != -100:
                return i  # 첫 번째 유효한 토큰 = 정답 시작 위치

        # 방법 2 (fallback): response_template 패턴 탐색 (마스킹되지 않은 경우)
        template_len = len(self.response_template_ids)
        for i in range(len(input_list) - template_len, -1, -1):
            if input_list[i : i + template_len] == self.response_template_ids:
                return i + template_len

        return -1  # 못 찾은 경우

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        모델의 Logits 중 정답 토큰(1~5)에 해당하는 부분만 추출합니다.

        completion_only_loss=True 환경에서는 labels가 -100으로 마스킹되므로,
        -100이 아닌 첫 번째 토큰 위치를 정답 위치로 사용합니다.
        그 직전 위치의 logits가 정답을 예측하는 logits입니다.
        """
        logits = logits if not isinstance(logits, tuple) else logits[0]

        batch_size = logits.size(0)
        batch_logits = []

        for i in range(batch_size):
            # 정답 시작 위치 찾기
            response_start = self._find_response_start(labels[i])

            if response_start > 0:
                # response_start - 1 위치에서 정답 토큰 예측
                pred_pos = response_start - 1
                batch_logits.append(logits[i, pred_pos, self.logit_idx])
            else:
                # fallback: 못 찾은 경우 마지막에서 3번째 위치 사용
                batch_logits.append(logits[i, -3, self.logit_idx])

        return torch.stack(batch_logits)

    def compute_metrics(self, evaluation_result):
        """
        평가 결과에서 macro_f1과 accuracy를 계산합니다.

        completion_only_loss=True 환경에서는 labels가 마스킹되어 있으므로,
        -100이 아닌 첫 번째 토큰을 정답으로 사용합니다.
        """
        logits, labels = evaluation_result

        # 디버깅: 첫 번째 샘플의 labels 정보 출력 (1회만)
        if not self._debug_printed and len(labels) > 0:
            self._debug_printed = True
            first_label = (
                labels[0].tolist() if hasattr(labels[0], "tolist") else list(labels[0])
            )
            print(
                f"\n[DEBUG] labels type: {type(labels)}, shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}"
            )
            print(f"[DEBUG] First label length: {len(first_label)}")
            print(f"[DEBUG] First label (first 10): {first_label[:10]}")
            print(f"[DEBUG] First label (last 10): {first_label[-10:]}")
            # -100이 아닌 첫 번째 위치 찾기
            non_masked_idx = -1
            for i, v in enumerate(first_label):
                if v != -100:
                    non_masked_idx = i
                    break
            print(f"[DEBUG] First non-masked index: {non_masked_idx}")
            if non_masked_idx >= 0 and non_masked_idx < len(first_label):
                token_id = first_label[non_masked_idx]
                print(f"[DEBUG] Token ID at that index: {token_id}")
                if token_id != -100:
                    print(f"[DEBUG] Decoded: {repr(self.tokenizer.decode([token_id]))}")

        # 1. 라벨에서 정답 추출 (-100이 아닌 첫 번째 토큰)
        parsed_labels = []
        valid_indices = []

        for idx, label_seq in enumerate(labels):
            response_start = self._find_response_start(label_seq)
            if response_start >= 0 and response_start < len(label_seq):
                label_list = (
                    label_seq.tolist()
                    if hasattr(label_seq, "tolist")
                    else list(label_seq)
                )
                answer_token_id = label_list[response_start]
                # -100이면 스킵 (패딩)
                if answer_token_id == -100:
                    continue
                answer_str = self.tokenizer.decode([answer_token_id]).strip()
                if answer_str in self.int_output_map:
                    parsed_labels.append(self.int_output_map[answer_str])
                    valid_indices.append(idx)
                else:
                    # 1~5가 아닌 경우 (이상한 토큰)
                    continue

        # 디버깅: 결과 출력
        print(
            f"[DEBUG] valid_indices count: {len(valid_indices)}, parsed_labels sample: {parsed_labels[:5] if parsed_labels else 'empty'}"
        )

        if not valid_indices:
            return {
                "macro_f1": 0.0,
                "accuracy": 0.0,
                "valid_sample_count": 0,
            }

        # 2. 예측값 추출 (Argmax) - 유효한 샘플만
        # torch tensor를 numpy로 변환
        if isinstance(logits, torch.Tensor):
            logits_np = logits.cpu().numpy()
        else:
            logits_np = np.array(logits)

        probs = torch.nn.functional.softmax(torch.tensor(logits_np), dim=-1).numpy()
        predictions = np.argmax(probs, axis=-1)
        filtered_preds = np.array([predictions[i] for i in valid_indices])
        filtered_labels = np.array(parsed_labels)

        # 3. Macro F1-score 및 정확도 계산
        # -1 레이블(범위 밖)은 preds(0~4)와 매칭되지 않아 자동으로 오답 처리
        macro_f1 = f1_score(
            filtered_labels, filtered_preds, average="macro", zero_division=0
        )
        acc = float(np.mean(filtered_labels == filtered_preds))

        return {
            "macro_f1": float(macro_f1),
            "accuracy": acc,
            "valid_sample_count": len(valid_indices),
        }
