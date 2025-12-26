# prompts/templates.py

BASE_PROMPT_FORMAT: str = """지문:
{paragraph}

{question_content}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""
