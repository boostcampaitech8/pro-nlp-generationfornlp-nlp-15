# utils/logger.py
from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(save_dir: Path, *, level: int = logging.INFO) -> None:
    save_dir = Path(save_dir)
    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # 중복 방지
    if root.handlers:
        return

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)