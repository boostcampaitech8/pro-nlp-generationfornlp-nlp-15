from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name_or_path: str


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_path: Path


class LoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_seq_length: int = Field(..., gt=0)
    padding_side: Literal["left", "right"] = "right"

    # pad_token을 eos로 맞출지 (Gemma류에서 흔함)
    pad_to_eos: bool = True

    # chat_template를 주입할지 (필요하면 모델별로 선택)
    use_chat_template: bool = True
    chat_template_name: Literal["gemma"] | None = "gemma"


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path

    per_device_train_batch_size: int = Field(..., gt=0)
    per_device_eval_batch_size: int = Field(..., gt=0)
    gradient_accumulation_steps: int = Field(..., gt=0)

    learning_rate: float = Field(..., gt=0.0)
    lr_scheduler_type: Literal[
        "linear",
        "cosine",
    ]

    num_train_epochs: int = Field(..., gt=0)

    fp16: bool = False
    bf16: bool = False
    tf32: bool = False
    gradient_checkpointing: bool = False

    logging_steps: int = Field(..., ge=1)
    evaluation_strategy: Literal["no", "steps", "epoch"]
    save_strategy: Literal["no", "steps", "epoch"]
    save_total_limit: int = Field(..., ge=0)

    seed: int
    weight_decay: float = Field(..., ge=0.0)
    report_to: Literal["none", "wandb", "tensorboard"] = "none"


class WandBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str
    entity: str
    name: str | None = None


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    data: DataConfig

    lora: LoRAConfig | None = None
    tokenizer: TokenizerConfig
    training: TrainingConfig
    wandb: WandBConfig | None = None
