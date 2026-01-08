"""
80B 모델을 사용한 Verifier Agent

CSV 모드 및 기존 Primary Agent의 CoT를 검증하기 위한 전용 Verifier
"""
import asyncio
from openai import AsyncOpenAI

from ..utils.answer_parser import parse_answer_from_response


class Verifier80BAgent:
    """
    80B 모델을 사용한 Verifier Agent
    """
    def __init__(self, config: dict):
        verifier_config = config.get('verifier_80b', {})
        
        self.client = AsyncOpenAI(
            base_url=verifier_config.get('base_url'),
            api_key=verifier_config.get('api_key', 'EMPTY'),
            timeout=verifier_config.get('timeout', 180),
            max_retries=0,  # 수동 재시도 로직 사용
        )
        self.model_name = verifier_config.get('model_name', 'local_model')
        self.temperature = verifier_config.get('temperature', 0.0)
        self.max_tokens = verifier_config.get('max_tokens', 2048)
        
        self.system_prompt = """당신은 수능/공무원 시험 문제를 검토하는 전문가입니다.

[작성 규칙]
1. 각 선택지 검토는 2-3줄로 간결하게 작성하세요. 같은 내용을 반복하지 마세요.
2. 핵심 논리와 근거만 제시하세요. 불필요한 반복 설명을 피하세요.
3. 반드시 응답의 마지막 줄에 "정답: (숫자)" 형식으로 답을 출력하세요.

다른 모델의 추론 과정을 검토하고, 오류가 있다면 간결하게 지적하여 올바른 답을 제시해주세요."""
        
        self.user_prompt_template = """[지문]
{paragraph}

[질문]
{question}
{question_plus}

[선택지]
{choices}

[1차 모델의 추론]
{primary_cot}

위 추론에 대해 검토해주세요:
1. 논리적 오류나 사실 오류가 있는지 확인하세요.
2. 누락된 정보나 잘못된 판단이 있는지 확인하세요.
3. 각 선택지에 대한 분석이 올바른지 검토하세요 (각 선택지는 2-3줄로 간결히).
4. 최종적으로 올바른 답을 제시하세요.

검토 결과를 간결하고 핵심적으로 서술하고, 반드시 마지막 줄에 "정답: (숫자)" 형식으로 답하세요."""
    
    async def verify(
        self,
        paragraph: str,
        question: str,
        question_plus: str,
        choices: list,
        primary_cot: str,
        semaphore: asyncio.Semaphore,
        max_retries: int = 3,
    ) -> tuple[int, str]:
        """
        80B 모델로 검증합니다.
        
        Returns:
            (answer: int, response: str)
        """
        async with semaphore:
            # 선택지 텍스트 구성
            choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            
            # question_plus 처리
            if question_plus and str(question_plus).strip() not in ['', 'nan', 'None']:
                question_plus_text = f"\n<보기>\n{question_plus}"
            else:
                question_plus_text = ""
            
            # 프롬프트 구성
            user_message = self.user_prompt_template.format(
                paragraph=paragraph,
                question=question,
                question_plus=question_plus_text,
                choices=choices_text,
                primary_cot=primary_cot
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # 재시도 로직
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    
                    content = response.choices[0].message.content
                    answer = parse_answer_from_response(content, len(choices))
                    return answer, content
                    
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)  # 지수 백오프
                        continue
                    else:
                        # 모든 재시도 실패
                        error_msg = f"Error ({type(e).__name__}): {str(e)}"
                        return 0, error_msg
            
            return 0, f"Error: {last_error}"

