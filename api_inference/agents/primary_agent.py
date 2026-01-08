"""
Primary Agent: 기존 CoT 방식으로 문제 풀이

32B 모델을 사용하여 Chain of Thought 방식으로 문제를 해결합니다.
"""
import asyncio
from typing import Any

from ..utils.api_client import AsyncAPIClient
from ..utils.answer_parser import parse_answer_from_response
from ..prompts import QuestionType, format_question_message, create_messages

# 최대 재시도 횟수
MAX_RETRY = 5


class PrimaryAgent:
    """
    Primary Agent (32B CoT) 실행
    
    기존 CoT 방식으로 문제를 풀이하고, 파싱 실패 시 재시도합니다.
    """
    
    def __init__(
        self,
        client: AsyncAPIClient,
        system_prompt: str,
        use_type_specific_prompt: bool = True,
        use_cot: bool = False,
    ):
        self.client = client
        self.system_prompt = system_prompt
        self.use_type_specific_prompt = use_type_specific_prompt
        self.use_cot = use_cot
    
    async def solve(
        self,
        item: dict[str, Any],
        semaphore: asyncio.Semaphore,
        max_retries: int = MAX_RETRY,
    ) -> tuple[int, str]:
        """
        문제를 풀이합니다.
        
        Args:
            item: 문제 데이터 딕셔너리 (paragraph, question, question_plus, choices, question_type 등)
            semaphore: 동시 요청 제한용 세마포어
            max_retries: 최대 재시도 횟수
        
        Returns:
            (answer: int, raw_response: str)
            - answer: 추출된 정답 (1~5), 파싱 불가 시 0
            - raw_response: LLM 원본 응답 또는 모든 시도의 응답 결합
        """
        async with semaphore:
            # 문제 유형 가져오기
            question_type = item.get('question_type', QuestionType.DEFAULT)
            
            # 프롬프트 생성 (유형별/기본 프롬프트, CoT 여부)
            user_message = format_question_message(
                paragraph=item['paragraph'],
                question=item['question'],
                question_plus=item.get('question_plus', ''),
                choices_list=item['choices'],
                question_type=question_type if self.use_type_specific_prompt else QuestionType.DEFAULT,
                use_cot=self.use_cot
            )
            
            # 유형별 시스템 프롬프트 또는 기본 프롬프트
            if self.use_type_specific_prompt:
                messages = create_messages(user_message, question_type=question_type)
            else:
                messages = create_messages(user_message, self.system_prompt)
            
            raw_response = ""
            answer = 0
            all_responses = []  # 모든 시도의 응답 저장
            
            # 최대 max_retries 횟수만큼 시도
            for attempt in range(1, max_retries + 1):
                try:
                    response = await self.client.chat_completion(messages=messages)
                    all_responses.append(f"[Attempt {attempt}] {response}" if response else f"[Attempt {attempt}] NULL")
                    
                    # 응답에서 정답 파싱
                    answer = parse_answer_from_response(response, len(item['choices']))
                    
                    # 파싱 성공 시 루프 종료
                    if answer > 0:
                        raw_response = response
                        break
                        
                except Exception as e:
                    all_responses.append(f"[Attempt {attempt}] Error: {str(e)}")
            
            # 모든 시도 후에도 파싱 실패한 경우
            if answer == 0:
                raw_response = " | ".join(all_responses)
            
            return answer, raw_response

