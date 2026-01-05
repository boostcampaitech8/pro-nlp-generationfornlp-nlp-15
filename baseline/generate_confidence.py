import argparse
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

import sys
import os
# Add project root to sys.path
sys.path.append(os.getcwd())

from baseline.configs.load import load_config
from baseline.models.loader import load_for_infer
from common.utils.logger import setup_logging
from common.data.load_dataset import load_tokenized_qa_dataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="baseline/configs/gemma3.yaml")
    parser.add_argument("--output", type=str, default="confidence_results.csv")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # 1. Load Configuration
    config = load_config(args.config)
    setup_logging(Path(config.train.output_dir))
    log = logging.getLogger(__name__)
    log.info(f"Loaded config: {args.config}")

    # 2. Load Model & Tokenizer
    # Automatically use the checkpoint defined in config or fallback to final_adapter
    adapter_path = config.infer.adapter_path
    if not adapter_path:
        # Check if checkpoint-914 exists (user mentioned it), otherwise final_adapter
        ckpt_path = Path(config.train.output_dir) / "checkpoint-914"
        if ckpt_path.exists():
            adapter_path = ckpt_path
        else:
            adapter_path = Path(config.train.output_dir) / "final_adapter"
    
    log.info(f"Loading model with adapter: {adapter_path}")
    model, tokenizer = load_for_infer(config, adapter_path=adapter_path)
    model.eval()
    device = next(model.parameters()).device

    # 3. Load Test Dataset
    # CRITICAL: use_cot=False for Non-CoT extraction
    log.info(f"Loading test data from {config.infer.test_path} (use_cot=False)")
    ds = load_tokenized_qa_dataset(
        file_path=str(config.infer.test_path),
        tokenizer=tokenizer,
        max_length=config.tokenizer.max_seq_length,
        require_answer=False,
        # CRITICAL: add_generation_prompt=True ensures input ends with assistant start token
        # This is required to predict the immediate next answer token (1~5) with high confidence.
        # (Matched with inspect_outputs.py logic)
        add_generation_prompt=True, 
        enable_thinking=False,
        use_cot=False, 
        exclude_answer=True,
    )
    
    test_df = pd.read_csv(config.infer.test_path)
    ids = test_df["id"].astype(str).tolist()

    # 4. Inference Loop (Logits)
    choice_tokens = ["1", "2", "3", "4", "5"]
    choice_ids = tokenizer.convert_tokens_to_ids(choice_tokens)
    
    results = []
    
    log.info("Starting Logit Inference with Confidence Calculation...")
    
    with torch.inference_mode():
        for i, item in tqdm(enumerate(ds), total=len(ds), desc="Calculating Confidence"):
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            
            # Model Forward
            outputs = model(input_ids=input_ids)
            # Get logits of the last token
            last_token_logits = outputs.logits[0, -1, :].float()
            
            # Extract logits for 1,2,3,4,5
            target_logits = last_token_logits[choice_ids]
            
            # Softmax to get probabilities (Confidence)
            probs = torch.softmax(target_logits, dim=-1).cpu().numpy()
            
            # Prediction
            pred_idx = np.argmax(probs)
            pred_answer = str(pred_idx + 1)
            confidence = probs[pred_idx]
            
            # Record Data
            row = {
                "id": ids[i],
                "answer": pred_answer,
                "confidence": confidence,
                "prob_1": probs[0],
                "prob_2": probs[1],
                "prob_3": probs[2],
                "prob_4": probs[3],
                "prob_5": probs[4]
            }
            results.append(row)

    # 5. Save Results
    output_df = pd.DataFrame(results)
    output_path = Path(args.output)
    output_df.to_csv(output_path, index=False)
    
    log.info(f"Saved detailed confidence results to {output_path}")
    print(f"Successfully saved confidence results to: {output_path}")
    print(f"Average Confidence: {output_df['confidence'].mean():.4f}")

if __name__ == "__main__":
    main()
