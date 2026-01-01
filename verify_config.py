from baseline.configs.load import load_config
from pathlib import Path

try:
    config = load_config("baseline/configs/gemma3.yaml")
    print("✅ Configuration loaded successfully!")
    print(f"NEFTune Alpha: {config.train.neftune_noise_alpha}")
    print(f"RSLoRA: {config.train.use_rslora}")
    print(f"DoRA: {config.train.use_dora}")
except Exception as e:
    print(f"❌ Configuration failed to load: {e}")
