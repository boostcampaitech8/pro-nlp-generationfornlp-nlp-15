import pandas as pd
from ast import literal_eval
from datasets import Dataset
from model_utils import format_question_message

def load_and_preprocess_data(file_path, tokenizer, split_ratio=0.1):
    """
    데이터 로드, literal_eval 파싱, 동적 프롬프트 생성 및 토큰화를 수행합니다.
    """
    print(f"Loading and processing data from {file_path}...")
    df = pd.read_csv(file_path)

    # 1. 파싱 (literal_eval 적용)
    if isinstance(df['problems'].iloc[0], str):
        df['problems'] = df['problems'].apply(lambda x: literal_eval(x))
    
    # 2. 데이터 포맷팅
    processed_data = []
    for _, row in df.iterrows():
        p = row['problems']
        
        # model_utils의 함수를 사용하여 <보기> 유무가 반영된 메시지를 생성합니다.
        user_message = format_question_message(
            paragraph=row["paragraph"],
            question=p["question"],
            question_plus=p.get("question_plus"),
            choices_list=p["choices"]
        )

        # Chat Message 구조 생성
        processed_data.append({
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"{p['answer']}"}
            ]
        })

    # HF Dataset 변환
    dataset = Dataset.from_pandas(pd.DataFrame(processed_data))

    # 3. 토큰화 함수
    def tokenize_function(examples, tokenizer):
        # model_utils에서 이미 주입된 chat_template을 사용하여 텍스트 변환
        texts = [
            tokenizer.apply_chat_template(msg, tokenize=False) 
            for msg in examples["messages"]
        ]
        
        tokenized_output = tokenizer(
            texts,
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": tokenized_output["input_ids"],
            "attention_mask": tokenized_output["attention_mask"]
        }

    # 4. 맵핑 (Tokenize)
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names, 
        fn_kwargs={"tokenizer": tokenizer}
    )

    # 5. 길이 필터링 (1024 토큰 이하만 사용)
    original_len = len(tokenized_dataset)
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= 1024)
    print(f"Filtered {original_len - len(tokenized_dataset)} samples > 1024 tokens.")

    # 6. Train / Val 분리
    split_dataset = tokenized_dataset.train_test_split(test_size=split_ratio, seed=42)
    
    print(f"Final Train size: {len(split_dataset['train'])}, Val size: {len(split_dataset['test'])}")
    
    return split_dataset['train'], split_dataset['test']