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
    # 2. Load Model & Tokenizer (Non-CoT Model)
    # Using explicit path for Non-CoT model validation as requested
    adapter_path = Path("output_gemma_noncot/final_adapter")
    log.info(f"Loading Non-CoT model with adapter: {adapter_path}")
    
    model, tokenizer = load_for_infer(config, adapter_path=adapter_path)
    model.eval()
    device = next(model.parameters()).device

    # 3. Load Test Dataset (Augments - Unseen)
    data_path = "/data/ephemeral/home/ksat_data/augments.csv"
    log.info(f"Loading AUGMENTS data from {data_path} (use_cot=False)")
    
    # Load dataset with ground truth for accuracy
    from common.data.read_data import load_qa_examples_from_file
    raw_examples = load_qa_examples_from_file(data_path)
    
    # Map ID to Ground Truth Answer
    test_df = pd.read_csv(data_path)
    ids = test_df["id"].astype(str).tolist()
    
    # QAExample objects don't have 'id', but they are loaded in the same order as the CSV rows.
    gt_map = {str(curr_id): str(ex.answer) for curr_id, ex in zip(ids, raw_examples)}

    ds = load_tokenized_qa_dataset(
        file_path=str(data_path),
        tokenizer=tokenizer,
        max_length=config.tokenizer.max_seq_length,
        require_answer=True, # Valid data has answers
        add_generation_prompt=True, 
        enable_thinking=False,
        use_cot=False, 
        exclude_answer=True,
    )
    
    # 4. Inference Loop (Logits)
    choice_tokens = ["1", "2", "3", "4", "5"]
    choice_ids = tokenizer.convert_tokens_to_ids(choice_tokens)
    
    results = []
    incorrect_ids = []
    correct_count = 0
    
    log.info("Starting Logit Inference on Augments Set...")
    
    with torch.inference_mode():
        for i, item in tqdm(enumerate(ds), total=len(ds), desc="Evaluating"):
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            
            # Model Forward
            outputs = model(input_ids=input_ids)
            last_token_logits = outputs.logits[0, -1, :].float()
            
            # Extract logits & probs
            target_logits = last_token_logits[choice_ids]
            probs = torch.softmax(target_logits, dim=-1).cpu().numpy()
            
            # Prediction
            pred_idx = np.argmax(probs)
            pred_answer = str(pred_idx + 1)
            confidence = probs[pred_idx]
            
            # Ground Truth Check
            current_id = ids[i]
            gt_answer = str(gt_map.get(current_id, "N/A")).strip()
            
            is_correct = (pred_answer == gt_answer)
            if is_correct:
                correct_count += 1
            else:
                incorrect_ids.append(current_id)
            
            # Record Data
            row = {
                "id": current_id,
                "answer": pred_answer,
                "ground_truth": gt_answer,
                "is_correct": is_correct,
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
    output_path = Path(args.output).parent / "augments_confidence_results.csv"
    output_df.to_csv(output_path, index=False)
    
    # Save incorrect IDs for CoT analysis
    incorrect_path = Path(args.output).parent / "incorrect_ids_augments.txt"
    with open(incorrect_path, "w") as f:
        for iid in incorrect_ids:
            f.write(f"{iid}\n")
    
    accuracy = correct_count / len(ds)
    log.info(f"Accuracy: {accuracy:.4f} ({correct_count}/{len(ds)})")
    log.info(f"Saved detailed results to {output_path}")
    log.info(f"Saved {len(incorrect_ids)} incorrect IDs to {incorrect_path}")
    
    print(f"\n[Evaluation Complete]")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Incorrect predictions: {len(incorrect_ids)}")
    print(f"Saved trace to: {output_path}")
    print(f"Saved incorrect IDs to: {incorrect_path}")

if __name__ == "__main__":
    main()
