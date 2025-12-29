from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import yaml  # pip install pyyaml
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import KFold

def run_kfold_training(
    base_config_path: str | Path,
    kfold_pairs: list[tuple[Path, Path]],
    configs_out_dir: str | Path = "./configs_kfold",
    train_entry: list[str] | None = None,  # 예: ["python", "train.py", "--config"]
):
    base_config_path = Path(base_config_path)
    configs_out_dir = Path(configs_out_dir)
    configs_out_dir.mkdir(parents=True, exist_ok=True)

    if train_entry is None:
        train_entry = ["python", "train.py", "--config"]

    base = yaml.safe_load(base_config_path.read_text())

    base_out = Path(base["train"]["output_dir"])  # "./outputs_skt" 같은 값

    for i, (train_csv, valid_csv) in enumerate(kfold_pairs, start=1):
        cfg = deepcopy(base)

        fold_name = f"fold_{i:02d}"
        fold_out = base_out / fold_name

        cfg["train"]["train_path"] = str(train_csv)
        cfg["train"]["valid_path"] = str(valid_csv)
        cfg["train"]["output_dir"] = str(fold_out)

        # wandb run 이름도 fold별로 유니크하게
        if "wandb" in cfg and isinstance(cfg["wandb"], dict):
            base_name = cfg["wandb"].get("name", "run")
            cfg["wandb"]["name"] = f"{base_name}_{fold_name}"

        fold_cfg_path = configs_out_dir / f"{fold_name}.yaml"
        fold_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

        print(f"\n[RUN {fold_name}] config={fold_cfg_path}")
        subprocess.run([*train_entry, str(fold_cfg_path)], check=True)

def split_kfold_csv(
    input_csv: str | Path,
    output_dir: str | Path,
    n_splits: int = 5,
    seed: int = 42,
    shuffle: bool = True,
    prefix: str = "fold",
) -> List[Tuple[Path, Path]]:
    """
    Split a QA CSV into K train/validation CSV pairs (K-Fold).

    The original CSV format is preserved:
    columns = [id, paragraph, problems, question_plus]

    Output naming (consistent):
      {prefix}_{i:02d}_train.csv
      {prefix}_{i:02d}_valid.csv
    """

    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)

    out_paths: List[Tuple[Path, Path]] = []
    for i, (train_idx, valid_idx) in enumerate(kf.split(df), start=1):
        fold_name = f"{prefix}_{i:02d}"
        train_path = output_dir / f"{fold_name}_train.csv"
        valid_path = output_dir / f"{fold_name}_valid.csv"

        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)

        print(f"[Fold {i}/{n_splits} Done]")
        print(f"  train: {len(train_df)} samples → {train_path}")
        print(f"  valid: {len(valid_df)} samples → {valid_path}")

        out_paths.append((train_path, valid_path))

    return out_paths

# ---- 사용 ----
pairs = split_kfold_csv(
    "/data/ephemeral/home/ksat_data/balanced.csv",
    "/data/ephemeral/home/ksat_data/kfold",
    n_splits=5,
    seed=42,
)

run_kfold_training(
    base_config_path="./config/ax_light.yaml",
    kfold_pairs=pairs,
    configs_out_dir="./config/kfold",
    train_entry=["uv", "run", "python", "-m", "finetuning.train", "--config"],
)