import os


def set_wandb_env(
    *,
    project: str | None = None,
    entity: str | None = None,
    name: str | None = None,
    group: str | None = None,
    job_type: str | None = None,
    notes: str | None = None,
    override: bool = False,
) -> None:
    """
    W&B 환경 변수를 설정합니다.
    W&B SDK는 실행 시(wandb.init) 관련 환경 변수가 있으면 이를 우선적으로 사용합니다.
    """

    # 설정할 매핑 테이블 (파라미터 이름: 환경 변수 이름)
    env_mapping = {
        "project": "WANDB_PROJECT",
        "entity": "WANDB_ENTITY",
        "name": "WANDB_NAME",
        "group": "WANDB_RUN_GROUP",
        "job_type": "WANDB_JOB_TYPE",
        "notes": "WANDB_NOTES",
    }

    # str 형태를 받는 파라미터
    for param, env_var in env_mapping.items():
        value = locals()[
            param
        ]  # locals() 결과: 함수의 파라미터 이름이 key, 값이 value인 딕셔너리
        if value and (override or env_var not in os.environ):
            os.environ[env_var] = value

    # list 형태를 받는 tags 파라미터
    if tags and (override or "WANDB_TAGS" not in os.environ):
        os.environ["WANDB_TAGS"] = ",".join(tags)
