from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForMCQ:
    tokenizer: PreTrainedTokenizerBase
    max_length: int | None = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [f["prompt"] for f in features]
        answers = [f["answer"] for f in features]

        # 1) prompt tokenization
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # 2) labels 초기화
        labels = torch.full_like(input_ids, fill_value=-100)

        # 3) answer token id (single-token 강제)
        answer_token_ids = [
            self.tokenizer.encode(a, add_special_tokens=False)
            for a in answers
        ]

        for i, token_ids in enumerate(answer_token_ids):
            if len(token_ids) != 1:
                raise ValueError(
                    f"Answer must be single token, got {token_ids}"
                )

            answer_token_id = token_ids[0]
            prompt_len = attention_mask[i].sum().item()

            pos = prompt_len - 1
            input_ids[i, pos] = answer_token_id
            labels[i, pos] = answer_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }