from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from trl import SFTConfig, SFTTrainer
from transformers import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from ..configs.schema import Config
from common.tokenization.chat_template import GEMMA_CHAT_TEMPLATE

logger = logging.getLogger(__name__)


class SFTTrainingRunner:
    def __init__(
        self,
        *,
        config: Config,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Any,
        eval_dataset: Any,
        peft_config: Any | None = None,
        metrics: Any | None = None,
    ) -> None:
        if config.train is None:
            raise ValueError("config.training is required for training")

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.peft_config = peft_config
        self.metrics = metrics

        self._trainer: SFTTrainer | None = None

    def build_sft_config(self) -> SFTConfig:
        train = self.config.train
        tokenizer = self.config.tokenizer

        report_to = "wandb" if train.report_to == "wandb" else "none"
        run_name = self.config.wandb.name if self.config.wandb else None

        return SFTConfig(
            output_dir=str(train.output_dir),
            do_train=True,
            do_eval=True,
            num_train_epochs=train.num_train_epochs,
            learning_rate=float(train.learning_rate),
            per_device_train_batch_size=train.per_device_train_batch_size,
            per_device_eval_batch_size=train.per_device_eval_batch_size,
            gradient_accumulation_steps=train.gradient_accumulation_steps,
            lr_scheduler_type=train.lr_scheduler_type,
            weight_decay=train.weight_decay,
            logging_steps=train.logging_steps,
            save_strategy=train.save_strategy,
            eval_strategy=train.evaluation_strategy,
            save_total_limit=train.save_total_limit,
            fp16=train.fp16,
            bf16=train.bf16,
            tf32=train.tf32,
            gradient_checkpointing=train.gradient_checkpointing,
            max_length=tokenizer.max_seq_length,
            # [변경 1] 데이터셋이 Chat 포맷(List[Dict])이라면 text 필드 지정 불필요 (자동 감지)
            # 만약 데이터셋 컬럼명이 'messages'가 아니라면 dataset_text_field 또는 dataset_kwargs로 매핑 필요
            # dataset_text_field="text",
            report_to=report_to,
            run_name=run_name,
            save_only_model=True,
            seed=train.seed,
            # [변경 2] 이 옵션을 켜면 Chat Template을 분석해 Assistant 응답만 학습
            completion_only_loss=True,
        )

    def build_trainer(self) -> SFTTrainer:
        if self._trainer is not None:
            return self._trainer

        # tokenizer safety
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

        args = self.build_sft_config()

        # [변경 3] DataCollatorForCompletionOnlyLM 및 response_template 수동 지정 삭제
        # SFTTrainer가 tokenizer.chat_template과 completion_only_loss=True 설정을 보고
        # 자동으로 user 부분은 마스킹(-100), model 부분은 학습하도록 처리

        self._trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            compute_metrics=(self.metrics.compute_metrics if self.metrics else None),
            preprocess_logits_for_metrics=(
                self.metrics.preprocess_logits_for_metrics if self.metrics else None
            ),
            peft_config=self.peft_config,
            args=args,
        )
        return self._trainer

    def train(self) -> None:
        trainer = self.build_trainer()
        logger.info("Starting training...")
        trainer.train()

    def save_final(self, *, subdir: str = "final_adapter") -> Path:
        train = self.config.train
        out_dir = Path(train.output_dir) / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        trainer = self.build_trainer()
        trainer.model.save_pretrained(str(out_dir))
        self.tokenizer.save_pretrained(str(out_dir))

        logger.info("Saved final adapter to %s", out_dir)
        return out_dir
