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
        (User provided logic: logits[:, -2, logit_idx])
        """
        logits = logits if not isinstance(logits, tuple) else logits[0]

        # [Batch, Seq, Vocab] -> [Batch, 5] (특정 위치, 특정 후보군만 추출)
        # 주의: 이 로직은 입력 시퀀스의 끝이 [정답토큰, EOS] 형태라고 가정합니다.
        # 학습 데이터가 이 형태를 따르지 않으면 인덱스 에러나 잘못된 평가가 될 수 있습니다.
        logits = logits[:, -2, self.logit_idx]

        return logits

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
