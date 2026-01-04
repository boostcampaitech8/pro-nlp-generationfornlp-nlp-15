import argparse
import logging
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from .configs.load import load_config

from .utils.llm import build_llm

from common.utils.logger import setup_logging
from common.utils.wandb import set_wandb_env
# from common.data.load_dataset import load_qa_dataset_tokenized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    return parser.parse_args()


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
    
    # 6) inference
    llm = build_llm(config)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="요약해줘: 대규모 언어 모델(LLM)은 자연어 처리와 인공지능 연구의 핵심 요소로 자리잡고 있으며, 그 성능은 사용된 데이터셋에 크게 의존합니다. LLM의 성능을 극대화하기 위해서는 고품질의 데이터셋이 필수적이며, 다양한 도메인과 태스크에 적합한 데이터셋이 필요합니다. 하지만 지금까지 LLM에서 사용되는 데이터셋에 대한 체계적인 연구가 부족했던 상황입니다. 대규모 언어 모델을 위한 데이터셋: 종합적인 연구(Datasets for Large Language Models: A Comprehensive Survey) 라는 제목의 이 논문은 LLM 데이터셋을 다각도로 분석하고 분류하여 연구자와 개발자에게 중요한 참고자료를 제공합니다. 다양한 언어와 도메인에 걸쳐 방대한 양의 데이터를 다루며, 현재와 미래의 연구 방향성을 제시하고 있습니다"),
    ]

    res = llm.invoke(messages)
    print(res.content)
    
    # 6) metrics, data collator



if __name__ == "__main__":
    main()