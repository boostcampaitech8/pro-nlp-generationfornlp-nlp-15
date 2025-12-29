
import sys
import os
import random
import torch
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from baseline.configs.load import load_config
from baseline.models.loader import load_for_infer
from common.data.read_csv import load_qa_examples_from_csv
from common.data.message_builder import build_chat_messages
from transformers import TextStreamer

def main():
    # 1. Configuration
    config_path = "baseline/configs/gemma3.yaml"
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # 2. Load Model & Tokenizer (Unsloth via loader)
    print("Loading model and tokenizer using Unsloth...")
    # This calls load_for_infer which returns (model, tokenizer)
    # It internally handles FastLanguageModel.from_pretrained and .for_inference(model)
    model, tokenizer = load_for_infer(config)
    print("Model loaded successfully.")

    # 3. Load Dataset
    data_path = "/data/ephemeral/home/ksat_data/splitted/valid.csv"
    print(f"Loading data from {data_path}...")
    examples = load_qa_examples_from_csv(data_path)
    print(f"Total examples: {len(examples)}")

    # 4. Sample 1 random example
    samples = random.sample(examples, 1)

    print("\n" + "="*50)
    print(" Starting Inference Test on 1 Random Sample (Unsloth Generation Mode) ")
    print("="*50 + "\n")

    for i, example in enumerate(samples):
        print(f"--- Sample {i+1} ---")
        
        # Build messages
        # Use CoT prompt for Gemma 3 to enforce reasoning first
        messages_dict = build_chat_messages(example, use_cot=True)
        messages = messages_dict["messages"]
        
        # Apply chat template
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False, 
            )
        except TypeError:
            # Fallback if argument not accepted
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
        # Unsloth models are optimized, but standard generate kwargs should work
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            streamer=streamer,
        )

        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        print(f"[Generated Output]:\n{response}\n")
        print(f"[Ground Truth Answer]: {example.answer}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
