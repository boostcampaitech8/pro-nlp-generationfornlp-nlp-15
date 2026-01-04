from langchain_openai import ChatOpenAI

from ..configs.schema import Config

def build_llm(config: Config) -> ChatOpenAI:
    api = config.api
    inf = config.inference

    return ChatOpenAI(
        model=api.model_name,
        api_key=api.api_key,
        base_url=api.base_url,
        temperature=inf.temperature,
        max_tokens=inf.max_tokens,
        timeout=api.timeout,
    )