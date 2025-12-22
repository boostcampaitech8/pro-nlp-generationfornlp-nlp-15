from __future__ import annotations

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
    padding_side:  str,
    max_seq_length: int,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    tokenizer.padding_side = padding_side
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
    Load model + tokenizer for train (LoRA applied).
    """
    assert config.train is not None, "trainConfig required for train"

    torch_dtype = _resolve_torch_dtype(config)

    tokenizer = _load_tokenizer(
        config.model.name_or_path,
        padding_side=config.tokenizer.padding_side,
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
        padding_side=config.tokenizer.padding_side,
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
