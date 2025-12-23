import argparse
import logging
from pathlib import Path



from .configs.load import load_config
from .models.loader import load_for_train
from .trainer.metrics import CustomMetrics
from .trainer.sft_runner import SFTTrainingRunner
from .trainer.data_collator import DataCollatorForMCQ

from common.utils.logger import setup_logging
from common.utils.wandb import set_wandb_env
from common.data.load_dataset import load_qa_dataset_prompt_answer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
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
    set_wandb_env(
        project=wandb_conf.project,
        entity=wandb_conf.entity,
        name=wandb_conf.name,
        override=False,
    )

    # 4) model + tokenizer (+ LoRA)
    log.info("Loading model/tokenizer...")
    model, tokenizer, peft_config = load_for_train(config)

    # 5) dataset (single CSV -> train/val split)
    log.info("Building datasets...")
    train_ds = load_qa_dataset_prompt_answer(
        file_path=str(config.train.train_path),
        tokenizer=tokenizer,
        require_answer=True,
    )
    val_ds = load_qa_dataset_prompt_answer(
        file_path=str(config.train.valid_path),
        tokenizer=tokenizer,
        require_answer=False,
    )
    log.info("Dataset sizes: train=%d val=%d", len(train_ds), len(val_ds))

    # 6) metrics, data collator
    metrics = CustomMetrics(tokenizer)
    data_collator = DataCollatorForMCQ(
        tokenizer=tokenizer,
        max_length=tokenizer.model_max_length,
    )

    # 7) trainer runner
    runner = SFTTrainingRunner(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        metrics=metrics,
        data_collator=data_collator,
    )

    runner.train()
    runner.save_final()


if __name__ == "__main__":
    main()
