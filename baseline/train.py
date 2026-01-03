import argparse
import logging
from pathlib import Path


from .configs.load import load_config
from .models.loader import load_for_train
from .trainer.metrics import CustomMetrics
from .trainer.sft_runner import SFTTrainingRunner

from common.utils.logger import setup_logging
from common.utils.wandb import set_wandb_env
from common.data.load_dataset import load_text_qa_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="baseline/configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) config
    config = load_config(args.config)

    # 2) logging (실험 output_dir/logs/train.log)
    setup_logging(Path(config.train.output_dir))
    log = logging.getLogger(__name__)
    log.info("Loaded config: %s", args.config)

    # 3) set wandb
    wandb_conf = config.wandb
    if wandb_conf is not None:
        set_wandb_env(
            project=wandb_conf.project,
            entity=wandb_conf.entity,
            name=wandb_conf.name,
            group=wandb_conf.group,
            job_type=wandb_conf.job_type,
            notes=wandb_conf.notes,
            override=False,
        )

    # 4) model + tokenizer (+ LoRA)
    log.info("Loading model/tokenizer...")
    model, tokenizer, peft_config = load_for_train(config)

    # 5) dataset (single CSV -> train/val split OR separate files)
    log.info("Building datasets...")

    if config.train.eval_path:
        # Pre-split case
        log.info(f"Loading train from {config.train.train_path}")
        train_ds = load_text_qa_dataset(
            file_path=str(config.train.train_path),
            tokenizer=tokenizer,
            require_answer=True,
        )
        log.info(f"Loading valid from {config.train.eval_path}")
        val_ds = load_text_qa_dataset(
            file_path=str(config.train.eval_path),
            tokenizer=tokenizer,
            require_answer=True,
        )
    else:
        # Runtime stratified split case (for CoT training with single JSONL file)
        log.info(f"Loading train from {config.train.train_path} and stratified splitting")
        full_ds = load_text_qa_dataset(
            file_path=str(config.train.train_path),
            tokenizer=tokenizer,
            require_answer=True,
        )
        
        # Stratified split using sklearn
        from sklearn.model_selection import train_test_split
        
        indices = list(range(len(full_ds)))
        answers = full_ds["answer"]
        
        train_indices, val_indices = train_test_split(
            indices,
            test_size=0.1,
            stratify=answers,
            random_state=config.train.seed,
        )
        
        train_ds = full_ds.select(train_indices)
        val_ds = full_ds.select(val_indices)
        
        log.info(f"Stratified split: train={len(train_ds)}, val={len(val_ds)}")

    log.info("Dataset sizes: train=%d val=%d", len(train_ds), len(val_ds))

    # 6) metrics
    metrics = CustomMetrics(tokenizer)

    # 7) trainer runner
    runner = SFTTrainingRunner(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        metrics=metrics,
    )

    runner.train()
    runner.save_final()


if __name__ == "__main__":
    main()
