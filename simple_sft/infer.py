import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from functools import partial

from .configs.load import load_config
from .models.loader import load_for_infer

from common.utils.logger import setup_logging
from common.utils.wandb import set_wandb_env
from common.data.load_dataset import load_qa_dataset_tokenized
from common.data.chat_tokenizer import tokenize_prompt_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="baseline/configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) config
    config = load_config(args.config)

    # 2) logging
    setup_logging(Path(config.train.output_dir))
    log = logging.getLogger(__name__)
    log.info("Loaded config: %s", args.config)

    # 3) wandb env (optional)
    if config.wandb is not None:
        set_wandb_env(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.name,
            override=False,
        )

    # 4) model + tokenizer (+ adapter)
    adapter_path = None
    if config.infer.use_adapter:
        adapter_path = config.infer.adapter_path or (Path(config.train.output_dir) / "final_adapter")

    model, tokenizer = load_for_infer(config, adapter_path=adapter_path)
    model.eval()
    device = next(model.parameters()).device

    # 5) tokenized dataset + ids
    ds = load_qa_dataset_tokenized(
        file_path=str(config.infer.test_path),
        tokenizer=tokenizer,
        include_answer=False,
        max_length=config.tokenizer.max_seq_length
    )
    test_df = pd.read_csv(config.infer.test_path)
    ids = test_df["id"].astype(str).tolist()

    # 6) inference (next-token logits over "1"~"5")
    choice_ids = tokenizer.convert_tokens_to_ids(["1", "2", "3", "4", "5"])

    results: list[dict[str, str]] = []
    with torch.inference_mode():
        for i, item in tqdm(enumerate(ds), total=len(ds), desc="Inference", mininterval=0.5):
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long, device=device).unsqueeze(0)

            logits = model(input_ids=input_ids).logits[0, -1, :].float()
            target_logits = logits[choice_ids]
            probs = torch.softmax(target_logits, dim=-1).detach().cpu().numpy()
            pred = str(int(np.argmax(probs)) + 1)

            results.append({"id": ids[i], "answer": pred})

    # 7) save
    out_path = Path(config.infer.output_path)
    pd.DataFrame(results).to_csv(out_path, index=False)
    log.info("Saved: %s", out_path)
    print(f"[done] saved: {out_path}")


if __name__ == "__main__":
    main()