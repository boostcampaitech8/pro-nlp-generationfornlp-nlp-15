import warnings
from typing import Any, Optional, Union

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion-only language modeling.
    It ensures that the prompt part of the input is masked (label = -100).
    """

    def __init__(
        self,
        response_template: Union[str, list[int]],
        tokenizer: PreTrainedTokenizerBase,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, mlm=mlm, **kwargs)
        self.response_template = response_template
        self.ignore_index = ignore_index

        if isinstance(response_template, str):
            self.response_token_ids = self.tokenizer.encode(
                response_template, add_special_tokens=False
            )
        else:
            self.response_token_ids = response_template

    def torch_call(
        self, examples: list[Union[list[int], Any, dict[str, Any]]]
    ) -> dict[str, Any]:
        batch = super().torch_call(examples)

        # labels가 없으면 input_ids 복사해서 생성
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()

        for i in range(len(batch["labels"])):
            response_token_ids_start_idx = None

            # response_template 위치 찾기
            # (단순화를 위해 첫 번째 매칭만 찾음)
            for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                # response_token_ids 길이가 1인 경우
                if len(self.response_token_ids) == 1:
                    response_token_ids_start_idx = idx
                    break

                # 길이가 1보다 큰 경우 전체 매칭 확인
                if (
                    batch["labels"][
                        i, idx : idx + len(self.response_token_ids)
                    ].tolist()
                    == self.response_token_ids
                ):
                    response_token_ids_start_idx = idx
                    break

            if response_token_ids_start_idx is None:
                warnings.warn(
                    f"Could not find response key {self.response_template} in token IDs {batch['labels'][i]}"
                )
            else:
                # response 시작 인덱스 + response 길이까지 마스킹
                # 즉, User Prompt + Response Template 부분까지 -100 처리
                # (Response Template 자체도 프롬프트의 일부이므로 마스킹하는 것이 일반적)
                response_token_ids_end_idx = response_token_ids_start_idx + len(
                    self.response_token_ids
                )
                batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        return batch
