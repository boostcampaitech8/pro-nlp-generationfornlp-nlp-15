import torch
import numpy as np
import evaluate


class CustomMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # ✅ macro f1
        self.f1_metric = evaluate.load("f1")

        self.int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        self.logit_idx = self.tokenizer.convert_tokens_to_ids(["1", "2", "3", "4", "5"])

    def preprocess_logits_for_metrics(self, logits, labels):
        logits = logits if not isinstance(logits, tuple) else logits[0]
        return logits[:, -2, self.logit_idx]

    def compute_metrics(self, evaluation_result):
        logits, labels = evaluation_result

        # 1) labels decode
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 2) parse labels -> 0..4
        parsed_labels: list[int] = []
        for label in decoded_labels:
            clean_label = label.split("<end_of_turn>")[0].strip()
            parsed_labels.append(self.int_output_map.get(clean_label, 0))

        # 3) predictions (0..4)
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        # 4) macro F1
        f1 = self.f1_metric.compute(
            predictions=predictions,
            references=parsed_labels,
            average="macro",
        )
        return f1  # {"f1": ...}