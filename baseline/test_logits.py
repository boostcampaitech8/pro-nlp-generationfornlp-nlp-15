
import sys
import os
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from baseline.configs.load import load_config
from baseline.models.loader import load_for_infer
from common.data.read_csv import load_qa_examples_from_csv
from common.data.message_builder import build_chat_messages

def main():
    # 1. Configuration (hardcoded for quick testing, or use argparse)
    config_path = "baseline/configs/qwen3_32b.yaml"
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # Force logits mode for this test concept
    # (Though we are just calculating manually here)
    
    # 2. Load Model & Tokenizer
    print("Loading model and tokenizer using Unsloth...")
    model, tokenizer = load_for_infer(config)
    print("Model loaded successfully.")
    
    # 3. Load Dataset
    data_path = "/data/ephemeral/home/ksat_data/splitted/valid.csv"
    print(f"Loading data from {data_path}...")
    examples = load_qa_examples_from_csv(data_path)
    
    # 4. Sample 5 random examples
    samples = random.sample(examples, 5)
    
    print("\n" + "="*50)
    print(" Starting Logit Inspection Test (Next Token Prediction) ")
    print("="*50 + "\n")

    # Target tokens (answers are 1, 2, 3, 4, 5)
    # Note: Tokenization might involve spaces like " 1" or just "1" depending on tokenizer.
    # We will check both or stick to what commonly works.
    choice_tokens = ["1", "2", "3", "4", "5"]
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choice_tokens]
    
    # Verify choice ids
    print(f"Choice Tokens: {choice_tokens}")
    print(f"Choice IDs:    {choice_ids}")
    decoded_check = [tokenizer.decode([i]) for i in choice_ids]
    print(f"Decoded Check: {decoded_check}\n")

    for i, example in enumerate(samples):
        print(f"--- Sample {i+1} ---")
        
        # Build messages (NO CoT, NO Generation Prompt)
        # We just want the conversation history up to the user question
        messages_dict = build_chat_messages(example, use_cot=False) # system + user
        
        # CRITICAL RE-APPLY: Remove ground truth answer for inference testing
        messages = [m for m in messages_dict["messages"] if m["role"] != "assistant"]
        
        # Apply chat template
        # Try to disable thinking mode explicitly
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # FORCE DISABLE THINKING
            )
        except TypeError:
             # Fallback if tokenizer doesn't support 'enable_thinking' arg
             text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        print(f"[Input Context (End)]:\n...{text[-200:]}\n")

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model(**model_inputs)
            logits = outputs.logits[0, -1, :] # Last token logits
            
        # Extract logits for choices
        target_logits = logits[torch.tensor(choice_ids, device=model.device)]
        probs = torch.softmax(target_logits, dim=-1).cpu().numpy()
        
        # Display results
        print("[Probability Distribution]")
        results = []
        for choice, prob in zip(choice_tokens, probs):
            results.append((choice, prob))
            bar_len = int(prob * 50)
            bar = "█" * bar_len
            print(f" Choice {choice}: {prob:.4f}  {bar}")
            
        pred_idx = np.argmax(probs)
        pred_choice = choice_tokens[pred_idx]
        pred_prob = probs[pred_idx]
        
        print(f"\nPredicted Answer: {pred_choice} (Conf: {pred_prob:.4f})")
        print(f"Ground Truth:     {example.answer}")
        
        match = "CORRECT" if str(pred_choice) == str(example.answer) else "WRONG"
        print(f"Result: {match}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
