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
    # Smart adapter loading
    adapter_path = config.infer.adapter_path
    if not adapter_path:
        # Check for checkpoint-914 or others
        output_dir = Path(config.train.output_dir)
        checkpoints = sorted([d for d in output_dir.glob("checkpoint-*") if d.is_dir()], key=lambda x: int(x.name.split("-")[-1]))
        if checkpoints:
            adapter_path = checkpoints[-1]
            print(f"Found latest checkpoint: {adapter_path}")
        else:
            adapter_path = output_dir / "final_adapter"
    
    print(f"Loading model with adapter: {adapter_path}")
    model, tokenizer = load_for_infer(config, adapter_path=adapter_path)
    model.eval()
    
    # CRITICAL: Set padding side for generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. Load Dataset
    data_path = "/data/ephemeral/home/ksat_data/splitted/valid.csv"
    print(f"Loading data from {data_path}...")
    # Use load_qa_examples_from_file instead of read_csv for robustness
    examples = load_qa_examples_from_file(data_path)
    print(f"Total examples: {len(examples)}")

    # 4. Sample 1 random example
    samples = random.sample(examples, 1)

    print("\n" + "="*50)
    print(" Starting Inference Test on 1 Random Sample (CoT Generation Mode) ")
    print("="*50 + "\n")

    stopping_criteria = StoppingCriteriaList([AnswerStoppingCriteria(tokenizer)])

    for i, example in enumerate(samples):
        print(f"--- Sample {i+1} ---")
        
        # Build messages
        # Use CoT prompt for Gemma 3 to enforce reasoning first
        # For CoT testing, we explicitly set use_cot=True
        messages_dict = build_chat_messages(example, use_cot=True)
        # CRITICAL: Strip assistant message so we don't feed the answer to the model
        messages = [m for m in messages_dict["messages"] if m["role"] != "assistant"]
        
        print("\n[DEBUG] Messages dict before template:")
        for m in messages:
            print(f"Role: {m['role']}, Content (trunc): {m['content'][:100]}...")
        print("-" * 20)
        
        # Apply chat template
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True, # Enable thinking for CoT 
            )
        except TypeError:
            print("Warning: 'enable_thinking' arg not supported in apply_chat_template. Trying without it.")
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
        print(f"[Input Prompt]:\n{text}\n\n")

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Use TextStreamer for real-time output
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

        # Generate
        print("[Generating...]")
        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.7,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )

        print(f"\n[Ground Truth Answer]: {example.answer}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
