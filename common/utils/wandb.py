import os


def set_wandb_env(
    *,
    project: str | None = None,
    entity: str | None = None,
    name: str | None = None,
    override: bool = False,
) -> None:
    """
    Set WANDB_* environment variables.

    - override=False: already-set env vars are preserved
    - override=True : overwrite env vars with given values
    """
    if project and (override or "WANDB_PROJECT" not in os.environ):
        os.environ["WANDB_PROJECT"] = project

    if entity and (override or "WANDB_ENTITY" not in os.environ):
        os.environ["WANDB_ENTITY"] = entity

    if name and (override or "WANDB_NAME" not in os.environ):
        os.environ["WANDB_NAME"] = name
