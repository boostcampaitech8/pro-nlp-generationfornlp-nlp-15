import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .configs.load import load_config
from .models.loader import load_for_infer

from common.utils.logger import setup_logging
from common.utils.wandb import set_wandb_env
from common.data.load_dataset import load_tokenized_qa_dataset

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="baseline/configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) config
    config = load_config(args.config)

    # 2) logging
    setup_logging(Path(config.train.output_dir))
    log = logging.getLogger(__name__)
    log.info("Loaded config: %s", args.config)

    # 3) wandb config
    wandb_conf = config.wandb
    if wandb_conf is not None:
        set_wandb_env(
            project=wandb_conf.project,
            entity=wandb_conf.entity,
            name=wandb_conf.name,
            group=wandb_conf.group,
            job_type=wandb_conf.job_type,
            notes=wandb_conf.notes,
            override=False,
        )
        wandb.init()
    else:
        log.warning(
            "WandB settings not found in config. Skipping WandB initialization."
        )

    # 4) model + tokenizer (+ adapter)
    adapter_path = None
    if config.infer.use_adapter:
        adapter_path = config.infer.adapter_path or (
            Path(config.train.output_dir) / "final_adapter"
        )

        # adapter_path가 WandB Artifact 형식인 경우 다운로드(Artifact 형식: entity/project/name:version)
        if str(adapter_path).count("/") >= 2 and ":" in str(adapter_path):
            if wandb.run is None:
                wandb.init()
            log.info(f"Downloading adapter from WandB Artifact: {adapter_path}")
            artifact = wandb.use_artifact(str(adapter_path), type="model")
            adapter_path = artifact.download()
            log.info(f"Downloaded to: {adapter_path}")

    model, tokenizer = load_for_infer(config, adapter_path=adapter_path)
    model.eval()
    device = next(model.parameters()).device

    # 5) tokenized dataset + ids
    # Generation Mode requires adding the generation prompt
    is_generate_mode = getattr(config.infer, "inference_method", "logits") == "generate"
    
    ds = load_tokenized_qa_dataset(
        file_path=str(config.infer.test_path),
        tokenizer=tokenizer,
        max_length=config.tokenizer.max_seq_length,
        require_answer=False,
        add_generation_prompt=is_generate_mode,  # Pass generation prompt flag
        enable_thinking=is_generate_mode, # Enable thinking if generation mode
        use_cot=is_generate_mode, # Use CoT system prompt if generation mode (for Gemma 3 etc)
        exclude_answer=True, # CRITICAL: Always correct for valid/test sets to prevent feeding answer
    )
    test_df = pd.read_csv(config.infer.test_path)
    ids = test_df["id"].astype(str).tolist()

    results: list[dict[str, str]] = []
    
    # 6-A) Generation Mode
    if is_generate_mode:
        log.info(f"Running Inference in GENERATION mode (Thinking enabled if model supports it)")
        import re
        
        # Regex to find answer: Look for '정답', '답', 'Answer' followed by a number 1-5
        # We look for the last occurrence as the final conclusion
        answer_pattern = re.compile(r'(?:정답|답|Answer)\s*:?\s*.*?([1-5])')

        for i, item in tqdm(enumerate(ds), total=len(ds), desc="Gen-Inference"):
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            
            # Generate
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=getattr(config.infer, "max_new_tokens", 4096),
                    do_sample=True, # Thinking usually works best with sampling
                    temperature=0.7,
                )
            
            # Decode only new tokens
            new_tokens = generated_ids[0][len(input_ids[0]):]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Parse Answer
            matches = answer_pattern.findall(response)
            if matches:
                pred = matches[-1] # Take the last one found
            else:
                # Fallback: try to find just any single digit 1-5 at the very end of text
                # or just default to 1 if totally failed (rare if model follows instruction)
                simple_digit = re.findall(r'([1-5])', response)
                pred = simple_digit[-1] if simple_digit else "1"
            
            results.append({"id": ids[i], "answer": pred})

    # 6-B) Logits Mode (Original)
    else:
        log.info("Running Inference in LOGITS mode (Next Token Prediction)")
        choice_ids = tokenizer.convert_tokens_to_ids(["1", "2", "3", "4", "5"])
        
        with torch.inference_mode():
            for i, item in tqdm(
                enumerate(ds), total=len(ds), desc="Inference", mininterval=0.5
            ):
                input_ids = torch.tensor(
                    item["input_ids"], dtype=torch.long, device=device
                ).unsqueeze(0)

                logits = model(input_ids=input_ids).logits[0, -1, :].float()
                target_logits = logits[choice_ids]
                probs = torch.softmax(target_logits, dim=-1).detach().cpu().numpy()
                pred = str(int(np.argmax(probs)) + 1)

                results.append({"id": ids[i], "answer": pred})

    # 7) save
    out_path = Path(config.infer.output_path)
    pd.DataFrame(results).to_csv(out_path, index=False)
    log.info("Saved: %s", out_path)
    print(f"[done] saved: {out_path}")

    # 추론 결과 Artifact 업로드
    if wandb.run is not None:
        log.info("Uploading inference results to WandB Artifacts...")
        run_name = wandb.run.name.replace("/", "-")
        artifact = wandb.Artifact(
            name=f"{run_name or 'inference'}-result",
            type="result",
            description="Inference output CSV",
        )
        artifact.add_file(str(out_path))
        wandb.log_artifact(artifact)
        log.info("Result artifact uploaded.")
        wandb.finish()


if __name__ == "__main__":
    main()
