from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from trl import SFTConfig, SFTTrainer
from transformers import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from ..configs.schema import Config
from common.data.collator import DataCollatorForCompletionOnlyLM
from common.utils.template import get_response_template


import wandb

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
            report_to=report_to,
            run_name=run_name,
            save_only_model=True,
            seed=train.seed,
            # [변경] Unsloth 호환성을 위해 completion_only_loss 자동 기능을 끄고 수동 Collator 사용
            completion_only_loss=False,
            # 데이터셋의 텍스트 필드 지정 (formatting_func 결과가 저장될 가상의 필드)
            dataset_text_field="text",
            # 가장 좋은 모델 하나만 유지하기 위해 eval_loss를 기준 평가지표로 선정
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # save_strategy가 epoch/steps일 때 작동
        )

    def build_trainer(self) -> SFTTrainer:
        if self._trainer is not None:
            return self._trainer

        # tokenizer safety
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        args = self.build_sft_config()

        # [변경 3] DataCollatorForCompletionOnlyLM 및 response_template 수동 지정 삭제
        # SFTTrainer가 tokenizer.chat_template과 completion_only_loss=True 설정을 보고
        # 자동으로 user 부분은 마스킹(-100), model 부분은 학습하도록 처리

        # [변경] DataCollatorForCompletionOnlyLM 수동 설정
        # Unsloth는 formatting_func를 강제하므로, 이를 우회하면서 Masking을 하려면 Collator를 직접 써야 함
        # 토크나이저에서 response_template 자동 감지
        response_template = get_response_template(self.tokenizer)
        logger.info(f"Auto-detected response_template: {repr(response_template)}")

        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )

        # [변경] formatting_func 복구 (Unsloth 필수 요구사항)
        def formatting_prompts_func(examples):
            output_texts = []
            for prompt, completion in zip(examples["prompt"], examples["completion"]):
                output_texts.append(prompt + completion)
            return output_texts

        self._trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
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

        # WandB Artifact 업로드: eval_loss가 가장 낮았던 가중치를 업로드합니다.
        if (
            self.config.train.report_to == "wandb"
            and wandb is not None
            and wandb.run is not None
        ):
            logger.info("Uploading adapter to WandB Artifacts...")

            run_name = wandb.run.name.replace("/", "-")
            artifact = wandb.Artifact(
                name=f"{run_name or 'model'}-adapter",
                type="model",
                description="Final (best) LoRA adapter",
            )
            artifact.add_dir(str(out_dir))
            wandb.log_artifact(artifact)
            logger.info("Artifact uploaded successfully.")

        return out_dir
