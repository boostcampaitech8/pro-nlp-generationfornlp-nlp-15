import torch
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from ast import literal_eval
from model_utils import load_model_and_tokenizer, format_question_message

def run_inference():
    with open("configs/train_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 1. 모델 로드 
    model, tokenizer = load_model_and_tokenizer(config, is_train=False)
    model.eval()

    # 2. 데이터 로드
    test_df = pd.read_csv('data/test.csv')
    infer_results = []
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

    print("Starting Inference...")
    with torch.inference_mode():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            p = literal_eval(row['problems'])
            
            # 유틸리티 함수 사용하여 메시지 생성 (보기 유무 자동 처리)
            user_message = format_question_message(
                paragraph=row["paragraph"],
                question=p["question"],
                question_plus=p.get("question_plus"),
                choices_list=p["choices"]
            )

            messages = [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
            ]

            # 3. 토큰화 및 추론
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")

            logits = model(input_ids).logits[:, -1, :].flatten().cpu()
            
            # 선택지에 해당하는 토큰 점수만 추출
            target_ids = [tokenizer.vocab[str(i + 1)] for i in range(len(p["choices"]))]
            probs = torch.nn.functional.softmax(logits[target_ids], dim=-1).numpy()
            
            predict_value = pred_choices_map[np.argmax(probs)]
            infer_results.append({"id": row["id"], "answer": predict_value})

    pd.DataFrame(infer_results).to_csv("output.csv", index=False)
    print("Inference Complete!")

if __name__ == "__main__":
    run_inference()