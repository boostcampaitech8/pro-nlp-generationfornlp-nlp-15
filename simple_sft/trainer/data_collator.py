from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForMCQ:
    tokenizer: PreTrainedTokenizerBase
    max_length: int | None = None

    def __call__(self, features):
        if any("answer" not in f or f["answer"] is None for f in features):
            raise ValueError("DataCollatorForMCQ requires `answer` in all samples")
        
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        answers = [f["answer"] for f in features]

        # padding만 수행
        batch = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            return_tensors="pt",
        )

        labels = torch.full_like(batch["input_ids"], -100)

        for i, a in enumerate(answers):
            token_ids = self.tokenizer.encode(a, add_special_tokens=False)
            assert len(token_ids) == 1

            pos = batch["attention_mask"][i].sum().item() - 1
            labels[i, pos] = token_ids[0]

        batch["labels"] = labels
        return batch