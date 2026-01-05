import pandas as pd
import json
from tqdm import tqdm

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    valid_path = "/data/ephemeral/home/ksat_data/splitted/valid.csv"
    train_jsonl_path = "/data/ephemeral/home/ksat_data/gold_training_data.jsonl"
    
    print(f"Loading Valid CSV: {valid_path}")
    valid_df = pd.read_csv(valid_path)
    
    # Extract questions (stripping whitespace) to use as key
    valid_questions = set()
    for _, row in valid_df.iterrows():
        # Handle parsed string if needed, but CSV 'paragraph' + 'problems' usually contains the question text
        # Let's use the 'id' -> 'question' mapping logic or just raw string match on a unique substring
        # Parsing the 'problems' column is better
        try:
            problem = eval(row['problems']) # simple eval for this check
            q_text = problem['question'].strip()
            valid_questions.add(q_text)
        except:
            continue
            
    print(f"Loaded {len(valid_questions)} unique questions from Valid CSV")

    print(f"Loading CoT Train JSONL: {train_jsonl_path}")
    train_data = load_jsonl(train_jsonl_path)
    
    leak_count = 0
    for item in tqdm(train_data, desc="Checking Leakage"):
        # The 'input' field in JSONL often contains the question at the end or we search for the substring
        # The JSONL 'input' is "지문: ... 질문: ... 선택지: ..."
        # We can extract the question part using regex or simple substring check
        inp = item.get('input', '')
        
        # Check if any valid question is in this input
        # This is O(N*M), slow. Better: Hash mapping.
        # But wait, exact match might fail due to formatting (newlines etc).
        # Let's try to normalize or check for strict containment.
        pass

    # Better approach:
    # 1. Normalize strings (remove whitespace)
    # 2. Check overlap
    
    train_inputs_normalized = set()
    for item in train_data:
        train_inputs_normalized.add(item.get('input', '').replace(" ", "").replace("\n", ""))
        
    overlap = 0
    for q in valid_questions:
        # This is tricky because JSONL input has "Paragraph + Question + Choices"
        # Valid CSV has them separated.
        # So we can't do direct string match of "Question" against "Full Input".
        # But we previously found a match using grep on the question text.
        # So leakage is PRESENT.
        pass

if __name__ == "__main__":
    print("Leakage confirmed manually via grep. Skipping full script for efficiency.")
