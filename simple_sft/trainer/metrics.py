import torch
import numpy as np
import evaluate
from transformers import PreTrainedTokenizerBase

def debug_decode_logits(
    tokenizer,
    logits: np.ndarray,   # (T, V) for one sample
    labels: np.ndarray,   # (T,)
    top_k: int = 20,
):
    # 1. answer position 찾기
    pos = np.where(labels != -100)[0]
    if len(pos) != 1:
        print("❌ invalid answer position:", pos)
        return

    p = pos[0]
    print(f"answer position: {p}")

    # 2. 정답 토큰
    gold_token_id = int(labels[p])
    print(
        "gold token:",
        gold_token_id,
        repr(tokenizer.decode([gold_token_id]))
    )

    # 3. 해당 위치의 vocab logits
    vocab_logits = logits[p]   # (V,)

    # 4. TOP-K
    topk_ids = np.argsort(vocab_logits)[-top_k:][::-1]

    print(f"\n=== TOP {top_k} LOGITS ===")
    for rank, tid in enumerate(topk_ids):
        token = tokenizer.decode([int(tid)])
        score = vocab_logits[tid]
        print(f"{rank:02d} | id={tid:<6} | logit={score:8.3f} | token={repr(token)}")
        
class CustomMetrics:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.f1_metric = evaluate.load("f1")
        self.accuracy_metric = evaluate.load("accuracy")

        # The specific tokens used in your MCQ (1-5)
        self.target_tokens = ["1", "2", "3", "4", "5"]
        self.logit_token_ids = tokenizer.convert_tokens_to_ids(self.target_tokens)
        
        # Map token_id -> class index (0-4)
        self.token_id_to_class = {tid: i for i, tid in enumerate(self.logit_token_ids)}

    def preprocess_logits_for_metrics(self, logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]

        # (B, T, V) → (B, T, 5)
        return logits[:, :, self.logit_token_ids]
    
    def compute_metrics(self, evaluation_result):
        logits, labels = evaluation_result

        # logits: (B, T, 5)  ← preprocess_logits_for_metrics에서 이미 5-class로 줄였다고 가정
        # labels: (B, T)

        logits = np.asarray(logits)
        labels = np.asarray(labels)

        # 정답 위치만 선택 (label != -100)
        mask = labels != -100

        # active_logits: (N, 5)
        # active_labels: (N,)
        active_logits = logits[mask]
        active_labels = labels[mask]

        if active_labels.size == 0:
            raise ValueError("No active labels found for metric computation.")

        # 예측: 5-class argmax
        preds = np.argmax(active_logits, axis=-1)

        # 정답: token_id → class index (0~4)
        refs = np.fromiter(
            (self.token_id_to_class[int(tid)] for tid in active_labels),
            dtype=np.int64,
            count=active_labels.shape[0],
        )

        f1 = self.f1_metric.compute(
            predictions=preds,
            references=refs,
            average="macro",
        )

        acc = self.accuracy_metric.compute(
            predictions=preds,
            references=refs,
        )

        return {
            "f1_macro": f1["f1"],
            "accuracy": acc["accuracy"],
        }