# prompts/system.py

SYSTEM_PROMPT: str = "지문을 읽고 질문의 답을 구하세요."

SYSTEM_PROMPT_COT: str = (
    "지문을 읽고 질문의 답을 구하세요. "
    "정답을 선택하기 전에 논리적인 추론 과정을 단계별로 서술하세요. "
    "마지막에 '정답: X' 형식으로 결론을 내리세요."
)