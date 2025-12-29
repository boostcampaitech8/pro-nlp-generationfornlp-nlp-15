from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from peft import LoraConfig, PeftModel, get_peft_model

from unsloth import FastLanguageModel

from ..configs.schema import Config


logger = logging.getLogger(__name__)

def _resolve_torch_dtype(cfg: Config) -> torch.dtype:
    train = cfg.train
    if train is not None:
        if train.bf16:
            return torch.bfloat16
        if train.fp16:
            return torch.float16
    return torch.float32


def _normalize_tokenizer(tok, *, max_seq_length: int) -> PreTrainedTokenizerBase:
    # Gemma3Processor 같은 경우 내부 tokenizer로 교체
    if hasattr(tok, "tokenizer"):
        tok = tok.tokenizer

    if not tok.pad_token:
        tok.pad_token = tok.eos_token

    tok.padding_side = "right"
    tok.model_max_length = max_seq_length
    return tok


def _load_base(
    cfg: Config,
    *,
    use_unsloth: bool,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """base model + tokenizer 로드 (unsloth / normal 공통 인터페이스)"""
    if use_unsloth:
        model, tok = FastLanguageModel.from_pretrained(
            model_name=cfg.model.name_or_path,
            max_seq_length=cfg.tokenizer.max_seq_length,
            dtype=_resolve_torch_dtype(cfg),
            load_in_4bit=cfg.model.load_in_4bit,
        )
        tok = _normalize_tokenizer(tok, max_seq_length=cfg.tokenizer.max_seq_length)
        return model, tok

    tok = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    tok = _normalize_tokenizer(tok, max_seq_length=cfg.tokenizer.max_seq_length)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        torch_dtype=_resolve_torch_dtype(cfg),
        device_map="auto",
    )
    return model, tok


def _apply_lora_for_train(
    cfg: Config,
    model: AutoModelForCausalLM,
    *,
    use_unsloth: bool,
) -> Tuple[AutoModelForCausalLM, Optional[LoraConfig]]:
    """train용 LoRA 적용 (unsloth면 lora_cfg=None, normal이면 lora_cfg 반환)"""
    assert cfg.train is not None

    if use_unsloth:
        model = FastLanguageModel.get_peft_model(
            model=model,
            r=cfg.train.lora_r,
            lora_alpha=cfg.train.lora_alpha,
            lora_dropout=cfg.train.lora_dropout,
            target_modules=cfg.train.lora_target_modules,
            bias="none",
            # Unsloth는 보통 True 대신 "unsloth"를 권장 패턴으로 씀
            use_gradient_checkpointing=("unsloth" if cfg.train.gradient_checkpointing else False),
            random_state=cfg.train.seed,
        )
        return model, None

    # normal(HF+PEFT)
    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=cfg.train.lora_r,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        target_modules=cfg.train.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model, lora_cfg


def load_for_train(
    cfg: Config,
    *,
    adapter_path: Path | None = None,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase, Optional[LoraConfig]]:
    """
    Train: base load -> LoRA apply (unsloth 지원)
    """
    use_unsloth = bool(getattr(cfg.model, "use_unsloth", False))

    model, tok = _load_base(cfg, use_unsloth=use_unsloth)
    model, lora_cfg = _apply_lora_for_train(cfg, model, use_unsloth=use_unsloth)
    
    if adapter_path is not None:
        logger.info("Loading adapter for continued training: %s", str(adapter_path))
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("trainable ratio: %.6f", trainable / total)

    return model, tok, lora_cfg


def load_for_infer(
    cfg: Config,
    *,
    adapter_path: Path | None = None,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """
    Infer: base load -> (unsloth면 for_inference) -> optional adapter load
    """
    use_unsloth = bool(getattr(cfg.model, "use_unsloth", False))

    model, tok = _load_base(cfg, use_unsloth=use_unsloth)

    if use_unsloth:
        FastLanguageModel.for_inference(model)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tok