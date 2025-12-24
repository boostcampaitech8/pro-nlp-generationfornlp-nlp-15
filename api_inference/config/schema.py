from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

# 1. API 접속 설정
class APIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    base_url: str
    api_key: str = Field("EMPTY")
    model_name: str = Field("local_model")
    timeout: int = Field(120, gt=0)
    max_retries: int = Field(3, ge=0)

# 2. 데이터 경로 설정
class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    train_path: Path
    test_path: Path
    output_dir: Path

# 3. LLM 추론 상세 파라미터
class InferenceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(1024, gt=0)
    seed: int | None = 42
    
    system_prompt: str
    
    max_concurrent: int = Field(5, gt=0)
    tool_choice: Literal["auto", "required", "none"] = "auto"
    use_type_specific_prompt: bool = False
    use_cot: bool = False
    use_streaming_save: bool = False
    save_batch_size: int = Field(20, gt=0)

# 4. W&B 설정
class WandBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str | None = None
    entity: str | None = None
    name: str | None = None

# 5. 전체 설정
class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    mode: Literal["train", "test"] = "test"
    
    api: APIConfig
    data: DataConfig
    inference: InferenceParams
    wandb: WandBConfig | None = None