from __future__ import annotations
from unsloth import FastLanguageModel

from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)

from ..configs.schema import Config


def _resolve_torch_dtype(cfg: Config) -> torch.dtype:
    if cfg.train is not None:
        if cfg.train.bf16:
            return torch.bfloat16
        if cfg.train.fp16:
            return torch.float16
    return torch.float32


def _load_tokenizer(
    model_name_or_path: str,
    *,
    max_seq_length: int,
    add_special_tokens: bool = True,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_seq_length

    return tokenizer


def _load_base_model(
    model_name_or_path: str,
    *,
    torch_dtype: torch.dtype,
) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )


def load_for_train(
    config: Config,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """
    Load model + tokenizer for train (LoRA applied, add unsloth).
    """
    assert config.train is not None, "trainConfig required for train"

    # Unsloth version
    if config.train.use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model.name_or_path,
            max_seq_length=config.tokenizer.max_seq_length,
            dtype=_resolve_torch_dtype(config),
            load_in_4bit=False,  # [비교 실험] Baseline(FP16)과 동일 조건 비교를 위해 4bit 비활성화
        )

        # Lora
        model = FastLanguageModel.get_peft_model(
            model=model,
            r=config.train.lora_r,
            lora_alpha=config.train.lora_alpha,
            lora_dropout=config.train.lora_dropout,
            target_modules=config.train.lora_target_modules,
            bias="none",
            use_gradient_checkpointing=config.train.gradient_checkpointing,  # !
            random_state=config.train.seed,
        )

        tokenizer.padding_side = "right"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer, None

    # Normal version
    else:
        torch_dtype = _resolve_torch_dtype(config)

        tokenizer = _load_tokenizer(
            config.model.name_or_path,
            max_seq_length=config.tokenizer.max_seq_length,
        )

        model = _load_base_model(
            config.model.name_or_path,
            torch_dtype=torch_dtype,
        )

        if config.train.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        lora_cfg = LoraConfig(
            r=config.train.lora_r,
            lora_alpha=config.train.lora_alpha,
            lora_dropout=config.train.lora_dropout,
            target_modules=config.train.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_cfg)
        return model, tokenizer, lora_cfg


def load_for_infer(
    config: Config,
    *,
    adapter_path: Path | None = None,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """
    Load model + tokenizer for inference.
    """
    torch_dtype = _resolve_torch_dtype(config)

    tokenizer = _load_tokenizer(
        config.model.name_or_path,
        max_seq_length=config.tokenizer.max_seq_length,
    )

    model = _load_base_model(
        config.model.name_or_path,
        torch_dtype=torch_dtype,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer
