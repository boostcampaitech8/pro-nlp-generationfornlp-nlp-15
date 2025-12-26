from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from trl import SFTConfig, SFTTrainer
from transformers import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
import torch.nn.functional as F
import torch
import csv
from pathlib import Path

from .callbacks import EvalPredictCallback

from ..configs.schema import Config

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
        data_collator: Any | None = None, 
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
        self.data_collator = data_collator

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
            warmup_steps=500,
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
            dataset_kwargs={"skip_prepare_dataset": True},
            completion_only_loss=True,
            report_to=report_to,
            run_name=run_name,
            save_only_model=True,
            seed=train.seed,
            remove_unused_columns=False,
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
        
        callbacks = [
            EvalPredictCallback(self)
        ]

        self._trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=(self.metrics.compute_metrics if self.metrics else None),
            preprocess_logits_for_metrics=(
                self.metrics.preprocess_logits_for_metrics if self.metrics else None
            ),
            peft_config=self.peft_config,
            callbacks=callbacks,
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


    def run_eval_prediction(self, trainer, tag: str):
        """
        현재 모델 상태로 eval_dataset 전체를 predict하고
        문제 단위 CSV 로그를 저장한다.
        - softmax 확률 포함
        - sample_id / gold / pred / correctness / prompt 기록
        """

        logger.info("[EvalPredict] running eval prediction (%s)", tag)

        # 1️⃣ eval 전체 prediction
        output = trainer.predict(self.eval_dataset)

        # ⚠️ preprocess_logits_for_metrics 적용된 logits임
        # shape: (N, T, num_classes=5)
        logits = torch.tensor(output.predictions)
        labels = torch.tensor(output.label_ids)

        sample_ids = self.eval_dataset["sample_id"]
        input_ids = self.eval_dataset["input_ids"]

        rows = []

        for i in range(len(sample_ids)):
            label_row = labels[i]

            # 2️⃣ 정답 위치 찾기 (label != -100)
            pos = (label_row != -100).nonzero(as_tuple=True)[0]
            if len(pos) != 1:
                continue
            p = pos.item()

            # 3️⃣ class logits → softmax 확률
            class_logits = logits[i, p]          # (5,)
            probs = F.softmax(class_logits, dim=-1)

            pred = int(torch.argmax(probs).item())

            # 4️⃣ gold label
            gold_token_id = int(label_row[p].item())
            gold = self.metrics.logit_token_ids.index(gold_token_id)

            # 5️⃣ prompt 디코딩 (answer 직전까지만)
            prompt = self.tokenizer.decode(
                input_ids[i][:p],
                skip_special_tokens=False,
            )

            # 6️⃣ row 구성
            rows.append([
                sample_ids[i],
                gold,
                pred,
                gold == pred,
                *[round(float(p), 6) for p in probs],  # p1~p5
                prompt,
            ])

        # 7️⃣ CSV 저장
        out_path = Path(self.config.train.output_dir) / f"eval_{tag}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_id",
                "gold",
                "pred",
                "correct",
                "prob_1",
                "prob_2",
                "prob_3",
                "prob_4",
                "prob_5",
                "prompt",
            ])
            writer.writerows(rows)

        logger.info(
            "[EvalPredict] saved %d rows to %s",
            len(rows),
            out_path,
        )