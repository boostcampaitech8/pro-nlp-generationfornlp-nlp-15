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


GEMMA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}"
    "{% if system_message is defined %}{{ system_message }}{% endif %}"
    "{% for message in messages %}{% set content = message['content'] %}"
    "{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
)


def _resolve_torch_dtype(cfg: Config) -> torch.dtype:
    if cfg.training is not None:
        if cfg.training.bf16:
            return torch.bfloat16
        if cfg.training.fp16:
            return torch.float16
    return torch.float32


def _load_tokenizer(
    model_name_or_path: str,
    *,
    max_seq_length: int,
    padding_side: str,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
    tokenizer.pad_token = tokenizer.eos_token
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
    Load model + tokenizer for training (LoRA applied).
    """
    assert config.training is not None, "TrainingConfig required for training"
    assert config.lora is not None, "LoRAConfig required for training"

    torch_dtype = _resolve_torch_dtype(config)

    tokenizer = _load_tokenizer(
        config.model.name_or_path,
        max_seq_length=config.tokenizer.max_seq_length,
        padding_side=config.tokenizer.padding_side,
    )

    model = _load_base_model(
        config.model.name_or_path,
        torch_dtype=torch_dtype,
    )

    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )

    model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def load_for_infer(
    config: Config,
    *,
    adapter_path: Path | None = None,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """
    Load model + tokenizer for inference.
    """
    tokenizer = _load_tokenizer(
        config.model.name_or_path,
        max_seq_length=config.tokenizer.max_seq_length,
        padding_side=config.tokenizer.padding_side,
    )

    model = _load_base_model(
        config.model.name_or_path,
        torch_dtype=torch.bfloat16,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer
