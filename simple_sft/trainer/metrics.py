import torch
import numpy as np
import evaluate


class CustomMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.f1_metric = evaluate.load("f1")

        self.int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        self.logit_token_ids = tokenizer.convert_tokens_to_ids(
            ["1", "2", "3", "4", "5"]
        )

    def preprocess_logits_for_metrics(self, logits, labels):
        return logits[:, :, self.logit_token_ids]

    def compute_metrics(self, evaluation_result):
        logits, labels = evaluation_result

        logits = np.asarray(logits)   # (B, T, V)
        labels = np.asarray(labels)   # (B, T)

        preds = []
        refs = []

        for i in range(labels.shape[0]):
            pos = np.where(labels[i] != -100)[0]
            assert len(pos) == 1
            p = pos[0]

            # logits → class space (5)
            class_logits = logits[i, p, :]
            pred = int(np.argmax(class_logits))

            # label token id → class index
            token_id = int(labels[i, p])
            ref = self.logit_token_ids.index(token_id)

            preds.append(pred)
            refs.append(ref)

        print("refs:", refs[:10])
        print("preds:", preds[:10])
        
        return self.f1_metric.compute(
            predictions=preds,
            references=refs,
            average="macro",
        )