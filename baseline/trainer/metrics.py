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

    def _find_response_start(self, input_ids) -> int:
        """
        input_ids에서 response_template의 끝 위치를 찾습니다.
        마지막 발생 위치를 찾아 그 끝 위치를 반환합니다.
        """
        template_len = len(self.response_template_ids)

        if isinstance(input_ids, torch.Tensor):
            input_list = input_ids.tolist()
        else:
            input_list = list(input_ids)

        # 마지막 발생 위치 찾기 (역순 탐색)
        for i in range(len(input_list) - template_len, -1, -1):
            if input_list[i : i + template_len] == self.response_template_ids:
                return i + template_len  # template 끝 위치 (정답 시작 위치)

        return -1  # 못 찾은 경우

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        모델의 Logits 중 정답 토큰(1~5)에 해당하는 부분만 추출합니다.

        response_template 기반 위치 탐색:
        - labels에서 <start_of_turn>model\n 이후 위치를 찾음
        - 그 직전 위치의 logits가 정답을 예측하는 logits임
        """
        logits = logits if not isinstance(logits, tuple) else logits[0]

        batch_size = logits.size(0)
        batch_logits = []

        for i in range(batch_size):
            # response_template 끝 위치 찾기
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
        User provided specific metric logic
        """
        logits, labels = evaluation_result

        # 1. 라벨에서 정답 추출 (response_template 이후 첫 토큰)
        parsed_labels = []
        for label_seq in labels:
            response_start = self._find_response_start(label_seq)
            if response_start > 0 and response_start < len(label_seq):
                answer_token_id = label_seq[response_start]
                answer_str = self.tokenizer.decode([answer_token_id]).strip()
                parsed_labels.append(self.int_output_map.get(answer_str, 0))
            else:
                parsed_labels.append(0)  # fallback

        # 2. 예측값 추출 (Argmax)
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        # 3. Macro F1-score 및 정확도 계산
        macro_f1 = f1_score(parsed_labels, predictions, average="macro")
        acc = self.acc_metric.compute(predictions=predictions, references=parsed_labels)

        return {
            "macro_f1": macro_f1,
            "accuracy": acc["accuracy"],
        }
