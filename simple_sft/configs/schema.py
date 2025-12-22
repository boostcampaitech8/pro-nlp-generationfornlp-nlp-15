from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name_or_path: str


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_seq_length: int = Field(..., gt=0)
    padding_side: Literal["left", "right"] = "right"

    # Gemma류: pad_token=eos_token으로 맞추는 플래그
    pad_to_eos: bool = True

    # chat_template 주입 여부/이름
    use_chat_template: bool = True
    chat_template_name: Literal["gemma"] | None = "gemma"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_path: Path
    output_dir: Path

    # LoRA (train only) - 1-depth 유지: nested 없이 train 내부에 flat로 둠
    lora_r: int = Field(..., gt=0)
    lora_alpha: int = Field(..., gt=0)
    lora_dropout: float = Field(..., ge=0.0, le=1.0)
    lora_target_modules: list[str]
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    lora_task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"

    # Trainer hyperparams
    per_device_train_batch_size: int = Field(..., gt=0)
    per_device_eval_batch_size: int = Field(..., gt=0)
    gradient_accumulation_steps: int = Field(..., gt=0)

    learning_rate: float = Field(..., gt=0.0)
    lr_scheduler_type: Literal["linear", "cosine"]

    num_train_epochs: int = Field(..., gt=0)

    fp16: bool = False
    bf16: bool = False
    tf32: bool = False
    gradient_checkpointing: bool = False

    logging_steps: int = Field(1, ge=1)
    evaluation_strategy: Literal["no", "steps", "epoch"] = "epoch"
    save_strategy: Literal["no", "steps", "epoch"] = "epoch"
    save_total_limit: int = Field(2, ge=0)

    seed: int = 42
    weight_decay: float = Field(0.0, ge=0.0)
    report_to: Literal["none", "wandb", "tensorboard"] = "none"


class InferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_path: Path
    output_path: Path = Path("output.csv")

    # adapter_path가 None이면 (train.output_dir / "final_adapter")로 코드에서 자동 결정
    use_adapter: bool = True
    adapter_path: Path | None = None

    # logits 방식이 깨질 때 fallback
    fallback_generate: bool = True
    max_new_tokens: int = Field(8, gt=0)


class WandBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str | None = None
    entity: str | None = None
    name: str | None = None


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    tokenizer: TokenizerConfig

    train: TrainConfig
    infer: InferConfig

    wandb: WandBConfig | None = None
