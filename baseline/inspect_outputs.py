import sys
import os
import random
import torch
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from baseline.configs.load import load_config
from baseline.models.loader import load_for_infer
from common.data.read_csv import load_qa_examples_from_csv
from common.data.message_builder import build_chat_messages

def main():
    # 1. Configuration matches recent training
    config_path = "baseline/configs/gemma3.yaml"
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)
    
    # 2. Load Model & Tokenizer
    # We will use the adapter we just trained
    adapter_path = Path(config.train.output_dir) / "final_adapter"
    
    print(f"Loading model with adapter from: {adapter_path}")
    model, tokenizer = load_for_infer(config, adapter_path=adapter_path)
    model.eval()
    
    # 3. Load Validation Dataset
    data_path = "/data/ephemeral/home/ksat_data/splitted/valid.csv"
    print(f"Loading data from {data_path}...")
    examples = load_qa_examples_from_csv(data_path)
    
    # 4. Sample 3 random examples
    samples = random.sample(examples, 3)
    
    print("\n" + "="*60)
    print(" [LOGIT INSPECTION] Checking Probabilities for Answer Tokens (1~5) ")
    print("="*60 + "\n")

    choice_tokens = ["1", "2", "3", "4", "5"]
    choice_ids = tokenizer.convert_tokens_to_ids(choice_tokens)
    
    for i, example in enumerate(samples):
        print(f"--- Sample {i+1} ---")
        print(f"Question (trunc): {example.question[:100]}...")
        
        # Build messages (Standard SFT format)
        messages_dict = build_chat_messages(example, use_cot=True)
        messages = [m for m in messages_dict["messages"] if m["role"] != "assistant"]
        
        # Apply chat template (generation prompt)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model(**model_inputs)
            # Last token logits (position where model should start answering)
            logits = outputs.logits[0, -1, :] 
            
        # 1. Check probabilities for 1~5
        target_logits = logits[torch.tensor(choice_ids, device=model.device)]
        probs = torch.softmax(target_logits, dim=-1).cpu().numpy()
        
        print("\n[Probabilities for 1-5 (Logits Mode)]")
        for c, p in zip(choice_tokens, probs):
            bar = "█" * int(p * 50)
            print(f" Choice {c}: {p:.5f}  {bar}")

        # 2. What did the model ACTUALLY want to say? (Top 5 tokens)
        print("\n[Top 5 Most Likely Next Tokens]")
        topk_vals, topk_indices = torch.topk(logits, 5)
        topk_probs = torch.softmax(logits, dim=-1)
        
        for idx, val in zip(topk_indices, topk_vals):
            token_str = tokenizer.decode([idx])
            prob = topk_probs[idx].item()
            print(f" Token '{token_str}' (ID:{idx.item()}): {prob:.5f}")
            
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
