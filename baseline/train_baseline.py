"""
Baseline Training Script (adapted from baseline_custom.ipynb)

This standalone script replicates the baseline notebook's training logic
for fair comparison with the modularized train.py approach.

Usage:
    uv run python -m baseline.train_baseline

Key features:
    - Implements custom labels masking (since DataCollatorForCompletionOnlyLM
      is not available in TRL 0.26.1)
    - Fixed logits index [-2] for metrics (same as baseline)
    - Same hyperparameters as baseline_custom.ipynb
"""

import os
import random
import argparse
from ast import literal_eval
from typing import Any

import torch
import numpy as np
import pandas as pd
import evaluate
from sklearn.metrics import f1_score

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = "/data/ephemeral/home/data"
OUTPUT_DIR = "./outputs_baseline"
MODEL_NAME = "beomi/gemma-ko-2b"

# Training hyperparameters (same as baseline_custom.ipynb)
CONFIG = {
    "seed": 42,
    "max_seq_length": 1024,
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "save_total_limit": 2,
    "logging_steps": 1,
}

# LoRA config (same as baseline_custom.ipynb)
LORA_CONFIG = {
    "r": 6,
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# Chat template (Gemma style)
GEMMA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}"
    "{% if system_message is defined %}{{ system_message }}{% endif %}"
    "{% for message in messages %}{% set content = message['content'] %}"
    "{% if message['role'] == 'user' %}{{ '<start_of_turn>user\\n' + content + '<end_of_turn>\\n<start_of_turn>model\\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\\n' }}{% endif %}{% endfor %}"
)

# Prompt templates
PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def flatten_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the problems column from JSON to separate columns."""
    records = []
    for _, row in df.iterrows():
        problems = literal_eval(row["problems"])
        record = {
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": problems.get("question"),
            "choices": problems.get("choices"),
            "answer": problems.get("answer"),
            "question_plus": problems.get("question_plus"),
        }
        records.append(record)
    return pd.DataFrame(records)


def format_chat_message(example: dict) -> dict:
    """Format a single example into chat message format."""
    choices_string = "\n".join(
        [f"{idx + 1} - {choice}" for idx, choice in enumerate(example["choices"])]
    )

    if example.get("question_plus"):
        user_message = PROMPT_QUESTION_PLUS.format(
            paragraph=example["paragraph"],
            question=example["question"],
            question_plus=example["question_plus"],
            choices=choices_string,
        )
    else:
        user_message = PROMPT_NO_QUESTION_PLUS.format(
            paragraph=example["paragraph"],
            question=example["question"],
            choices=choices_string,
        )

    return {
        "id": example["id"],
        "paragraph": example["paragraph"],
        "messages": [
            {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"{example['answer']}"},
        ],
        "label": example["answer"],
    }


# -----------------------------------------------------------------------------
# Custom Data Collator with Labels Masking
# -----------------------------------------------------------------------------
class DataCollatorForCompletionOnlyLM:
    """
    Custom data collator that masks labels before response_template.
    This replicates the functionality of trl's DataCollatorForCompletionOnlyLM
    which is not available in TRL 0.26.1.
    """

    def __init__(self, tokenizer, response_template: str, padding: bool = True):
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        self.padding = padding

    def __call__(self, features: list[dict]) -> dict:
        # Extract input_ids, attention_mask, and labels
        batch_input_ids = [f["input_ids"] for f in features]
        batch_attention_mask = [f["attention_mask"] for f in features]
        batch_labels = [f.get("labels", f["input_ids"]) for f in features]

        # Find max length for padding
        max_length = max(len(ids) for ids in batch_input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for input_ids, attention_mask, labels in zip(
            batch_input_ids, batch_attention_mask, batch_labels
        ):
            # Find response template position and mask labels
            response_start = self._find_response_start(input_ids)
            masked_labels = list(labels) if isinstance(labels, list) else labels.copy()

            if response_start > 0:
                for i in range(response_start):
                    masked_labels[i] = -100

            # Pad to max length
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                padded_input_ids.append(
                    input_ids + [self.tokenizer.pad_token_id] * padding_length
                )
                padded_attention_mask.append(attention_mask + [0] * padding_length)
                padded_labels.append(masked_labels + [-100] * padding_length)
            else:
                padded_input_ids.append(input_ids)
                padded_attention_mask.append(attention_mask)
                padded_labels.append(masked_labels)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    def _find_response_start(self, input_ids: list[int]) -> int:
        """Find the end position of response_template in input_ids."""
        template_len = len(self.response_template_ids)
        for i in range(len(input_ids) - template_len, -1, -1):
            if input_ids[i : i + template_len] == self.response_template_ids:
                return i + template_len
        return -1


# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default=f"{DATA_DIR}/train.csv")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument(
        "--report_to", type=str, default="none", choices=["none", "wandb"]
    )
    args = parser.parse_args()

    print(f"[INFO] Setting seed: {CONFIG['seed']}")
    set_seed(CONFIG["seed"])

    # =========================================================================
    # 1. Load and preprocess data
    # =========================================================================
    print(f"[INFO] Loading data from: {args.train_path}")
    train_df = flatten_dataset(pd.read_csv(args.train_path))
    print(f"[INFO] Total samples: {len(train_df)}")

    # Convert to Dataset and format messages
    dataset = Dataset.from_pandas(train_df)
    cols_to_remove = [
        col for col in dataset.column_names if col not in ["id", "paragraph"]
    ]
    dataset = dataset.map(
        format_chat_message,
        num_proc=4,
        remove_columns=cols_to_remove,
    )

    # =========================================================================
    # 2. Load model and tokenizer
    # =========================================================================
    print(f"[INFO] Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Add special tokens (same as baseline_custom.ipynb)
    special_tokens = {"additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Set chat template and pad token
    tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # =========================================================================
    # 3. Tokenize dataset
    # =========================================================================
    print("[INFO] Tokenizing dataset...")

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                tokenizer.apply_chat_template(example["messages"][i], tokenize=False)
            )
        return output_texts

    def tokenize(element):
        outputs = tokenizer(
            formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "paragraph": element["paragraph"],
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=4,
        load_from_cache_file=False,
        desc="Tokenizing",
    )

    # Train/eval split
    tokenized_dataset = tokenized_dataset.train_test_split(
        test_size=0.1, seed=CONFIG["seed"]
    )
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]

    print(
        f"[INFO] Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}"
    )

    # Token length statistics
    token_lengths = [len(x["input_ids"]) for x in train_dataset]
    print(
        f"[INFO] Token lengths - max: {max(token_lengths)}, min: {min(token_lengths)}, avg: {np.mean(token_lengths):.1f}"
    )

    # =========================================================================
    # 4. Setup metrics (baseline style - fixed index)
    # =========================================================================
    acc_metric = evaluate.load("accuracy")
    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
    logit_idx = [
        tokenizer.vocab["1"],
        tokenizer.vocab["2"],
        tokenizer.vocab["3"],
        tokenizer.vocab["4"],
        tokenizer.vocab["5"],
    ]

    def preprocess_logits_for_metrics(logits, labels):
        """Baseline approach: fixed index -2 (answer token before EOS)."""
        logits = logits if not isinstance(logits, tuple) else logits[0]
        # -2: answer token position (2nd from last)
        return logits[:, -2, logit_idx]

    def compute_metrics(evaluation_result):
        logits, labels = evaluation_result

        # Decode labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Parse labels (remove <end_of_turn> and map to int)
        parsed_labels = []
        for label in decoded_labels:
            clean_label = label.split("<end_of_turn>")[0].strip()
            parsed_labels.append(int_output_map.get(clean_label, 0))

        # Get predictions
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        # Calculate metrics
        macro_f1 = f1_score(parsed_labels, predictions, average="macro")
        acc = acc_metric.compute(predictions=predictions, references=parsed_labels)

        return {
            "macro_f1": macro_f1,
            "accuracy": acc["accuracy"],
        }

    # =========================================================================
    # 5. Setup training
    # =========================================================================
    peft_config = LoraConfig(**LORA_CONFIG)

    # Data collator with custom labels masking (same as baseline's DataCollatorForCompletionOnlyLM)
    response_template = "<start_of_turn>model"
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template,
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        max_length=CONFIG["max_seq_length"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        logging_steps=CONFIG["logging_steps"],
        save_strategy=CONFIG["save_strategy"],
        eval_strategy=CONFIG["evaluation_strategy"],
        save_total_limit=CONFIG["save_total_limit"],
        save_only_model=True,
        report_to=args.report_to,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        peft_config=peft_config,
        args=sft_config,
    )

    # =========================================================================
    # 6. Train!
    # =========================================================================
    print("[INFO] Starting training...")
    trainer.train()

    # Save final model
    final_path = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[INFO] Saved final adapter to: {final_path}")


if __name__ == "__main__":
    main()
