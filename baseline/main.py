import yaml
import argparse
from data_utils import load_and_preprocess_data
from model_utils import load_model_and_tokenizer
from trainer import run_training
from metrics import CustomMetrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 1. 모델과 토크나이저를 먼저 로드합니다. (데이터 처리에 토크나이저가 필요함)
    print("Loading model and tokenizer...")
    model, tokenizer, lora_config = load_model_and_tokenizer(config)

    # 2. 데이터 로드 (토크나이저 전달)
    train_dataset, val_dataset = load_and_preprocess_data(config['data']['train_path'], tokenizer)

    # 3. 평가 지표 객체 생성 (토크나이저 주입)
    metrics = CustomMetrics(tokenizer)

    # 4. 학습 시작
    run_training(model, tokenizer, train_dataset, val_dataset, config, metrics_obj=metrics, peft_config=lora_config)

if __name__ == "__main__":
    main()