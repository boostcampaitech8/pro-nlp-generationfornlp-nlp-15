import argparse
import logging
from pathlib import Path
from typing import Any
import csv
from tqdm import tqdm

from langchain_core.messages import SystemMessage, HumanMessage

from .configs.load import load_config

from .utils.llm import build_llm

from common.utils.logger import setup_logging
from common.utils.wandb import set_wandb_env
from common.data.read_csv import load_qa_examples_from_csv
from common.prompts.formatter import format_question_message

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)  # 같이 끄는 게 보통 좋음

SYSTEM_PROMPT = """
너는 CSAT(수능) 문항 분류기다.

입력: 지문/질문/선지
출력: 아래 형식의 '한 줄'만 출력한다.

형식(절대):
<subject> | <subcategory>

subject ∈ {국어, 사회탐구}

subcategory:
- subject=국어  -> {독서, 문학, 화법과 작문, 언어와 매체}
- subject=사회탐구 -> {생활과 윤리, 윤리와 사상, 한국지리, 세계지리, 동아시아사, 세계사, 경제, 정치와 법, 사회·문화}

규칙(핵심만):
- 역사/지리/윤리사상/경제/정치·법/사회현상 => 사회탐구
- 문학/문법/화법/작문 => 국어
- 사상가·학파·정의론·자연법/법실증주의 등 '이론 자체' 중심 => 윤리와 사상
- 헌법·선거·국가기관·기본권·형벌/절차·행정작용 등 '제도/법규' 중심 => 정치와 법
- 전쟁/혁명/제국/식민지/시대흐름 등 '역사 사건/사료' 중심 => 세계사(동아시아면 동아시아사)
- GDP/실업/물가/환율/IS-LM/탄력성/시장구조/효용 => 경제
- 사회조사/계층/문화/일탈/가족/교육 => 사회·문화

중요:
- 해설/추론과정/부가 텍스트 금지
- 출력은 정확히 한 줄만
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    return parser.parse_args()


def parse_pred(raw: str) -> tuple[str | None, str | None]:
    if not raw:
        return None, None
    s = raw.strip().replace("\n", " ")
    if "|" not in s:
        return None, None
    left, right = [x.strip() for x in s.split("|", 1)]
    return left or None, right or None


def main() -> None:
    args = parse_args()

    # 1) config
    config = load_config(args.config)

    # 2) logging (실험 output_dir/logs/train.log)
    setup_logging(Path(config.inference.output_dir))
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

    # 5) dataset (single CSV -> train/val split)
    log.info("Building datasets...")
    qas = load_qa_examples_from_csv(config.inference.data_path)

    # 6) inference
    llm = build_llm(config)

    # output path 결정
    out_dir = Path(getattr(config.inference, "output_dir", Path.cwd()))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir / "classification.csv")

    log.info(f"Running inference on {len(qas)} samples...")
    rows: list[dict[str, Any]] = []
    # 헤더 확정 (바로바로 쓰려면 미리 정해두는 게 안정적)
    fieldnames = ["id", "pred_raw", "subject", "subcategory", "error"]

    log.info(f"Streaming write to CSV: {out_path}")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for qa in tqdm(qas, desc="Classifying", unit="item"):
            user_input = format_question_message(
                qa.paragraph, qa.question, qa.question_plus, qa.choices
            )

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_input),
            ]

            row: dict[str, Any] = {
                "id": qa.id,
                "pred_raw": None,
                "subject": None,
                "subcategory": None,
                "error": None,
            }

            try:
                res = llm.invoke(messages)
                raw = (res.content or "").strip()
                subject, subcategory = parse_pred(raw)

                row["pred_raw"] = raw
                row["subject"] = subject
                row["subcategory"] = subcategory

            except Exception as e:
                row["error"] = repr(e)

            writer.writerow(row)
            f.flush()  # ✅ 매 샘플마다 디스크에 바로 반영


if __name__ == "__main__":
    main()
