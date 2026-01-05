import sys
import os
import random
import torch
import pandas as pd
import re
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from baseline.configs.load import load_config
from baseline.models.loader import load_for_infer
from common.data.read_data import load_qa_examples_from_file
from common.data.message_builder import build_chat_messages
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList

class AnswerStoppingCriteria(StoppingCriteria):
    """
    Custom stopping criteria to halt generation when the answer pattern is detected.
    Matches logic in infer.py
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Regex to match "Answer: [1-5]" pattern
        # Strict pattern: Newline + "정답/Answer" + Colon(:) + Number
        self.pattern = re.compile(r'(?:^|\n)\s*(?:정답|답|Answer)\s*:\s*([1-5])')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        matches = 0
        for text in decoded_texts:
            if self.pattern.search(text):
                matches += 1
        return matches == len(decoded_texts)

def main():
    # 1. Configuration
    config_path = "baseline/configs/gemma3.yaml"
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # 2. Load Model & Tokenizer
    # Explicitly use final_adapter as requested
    adapter_path = Path("outputs_gemma/final_adapter")
    print(f"Loading model with adapter: {adapter_path}")
    model, tokenizer = load_for_infer(config, adapter_path=adapter_path)
    model.eval()
    
    # CRITICAL: Set padding side for generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. Load Dataset
    data_path = "/data/ephemeral/home/ksat_data/augments.csv"
    print(f"Loading AUGMENTS data from {data_path}...")
    # Use load_qa_examples_from_file instead of read_csv for robustness
    examples = load_qa_examples_from_file(data_path)
    print(f"Total examples: {len(examples)}")

    # Load IDs from CSV to enable filtering by ID
    df = pd.read_csv(data_path)
    all_ids = df["id"].astype(str).tolist()
    
    # Create valid pairs of (example, id)
    # QAExample might be immutable/frozen, so we don't attach .id directly
    example_pairs = list(zip(examples, all_ids))

    # 4. Sample or Filter Logic
    incorrect_ids_path = Path("incorrect_ids_augments.txt")
    if incorrect_ids_path.exists():
        print(f"Found incorrect IDs file: {incorrect_ids_path}")
        with open(incorrect_ids_path, "r") as f:
            target_ids = set(line.strip() for line in f if line.strip())
        
        # Filter pairs where id is in target_ids
        filtered_pairs = [(ex, iid) for ex, iid in example_pairs if iid in target_ids]
        print(f"Filtered {len(filtered_pairs)} examples from incorrect list.")
        
        # PROCESS ALL FAILURES (No sampling)
        samples = filtered_pairs
    else:
        print("No incorrect_ids_augments.txt found. Using all data.")
        samples = example_pairs
    
    print("\n" + "="*50)
    print(f" Starting CoT Generation on {len(samples)} Items (BATCHED) ")
    print("="*50 + "\n")

    # BATCHING SETUP
    tokenizer.padding_side = "left" # Critical for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare inputs
    prompts = []
    metadata = []
    
    print("Preparing prompts...")
    for example, iid in samples:
        messages_dict = build_chat_messages(example, use_cot=True)
        messages = [m for m in messages_dict["messages"] if m["role"] != "assistant"]
        
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompts.append(text)
        metadata.append({"id": iid, "question": example.question, "ground_truth": example.answer})

    # Processing Loop
    batch_size = 4
    results = []
    
    # Simple batch iterator
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_meta = metadata[i:i+batch_size]
        
        print(f"Processing Batch {i//batch_size + 1}/{(len(prompts)//batch_size)+1} ...")

        # Tokenize batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                temperature=0.7,
                stopping_criteria=StoppingCriteriaList([AnswerStoppingCriteria(tokenizer)]),
            )
        
        # Decode batch
        input_len = inputs.input_ids.shape[1]
        new_tokens = generated_ids[:, input_len:]
        decoded_outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        for j, output_text in enumerate(decoded_outputs):
             meta = batch_meta[j]
             print(f"  > ID {meta['id']} generated {len(output_text)} chars")
             results.append({
                "id": meta["id"],
                "question": meta["question"],
                "ground_truth": meta["ground_truth"],
                "cot_output": output_text
            })

    # Save to CSV
    output_csv = "cot_failure_analysis.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved full analysis to {output_csv}")

if __name__ == "__main__":
    main()
