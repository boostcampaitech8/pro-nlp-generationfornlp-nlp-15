"""
Multi-Agent Processor: 전체 흐름 조율 (개선 버전)

기존 CoT 방식을 1차 Agent로 사용하고,
파싱 실패 시에만 Verifier Agent를 호출합니다.

Flow:
1. 1차 Agent: 기존 CoT 방식으로 문제 풀이
2. 답이 유효하면 (1~5) → 최종 답 반환
3. 답이 유효하지 않으면 → Verifier Agent 호출
4. Verifier: 재풀이 + CoT 검토 후 최종 답 결정
"""
import re
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

from api_inference.utils.api_client import AsyncAPIClient
from api_inference.prompts import QuestionType, format_question_message, create_messages
from .verifier_agent import VerifierAgent


# 파싱 재시도 횟수
MAX_RETRY = 3


@dataclass
class MultiAgentResult:
    """Multi-Agent 추론 결과"""
    id: str                     # 문제 ID
    final_answer: int           # 최종 답 (1-5, 또는 0=실패)
    answer: str                 # 최종 답 문자열 (기존 호환용)
    
    # 1차 Agent (기존 CoT) 결과
    primary_answer: int         # 1차 Agent의 답
    primary_raw_response: str   # 1차 Agent 원본 응답
    
    # Verifier 결과 (호출된 경우)
    verifier_used: bool         # Verifier 호출 여부
    verifier_method: str        # Verifier 사용 방법
    verifier_answer: int        # Verifier 최종 답
    verifier_raw_response: str  # Verifier 원본 응답
    
    # 메타 정보
    question_type: str          # 문제 유형
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)
    
    def to_output_dict(self) -> Dict[str, Any]:
        """출력용 딕셔너리 (기존 형식과 호환)"""
        return {
            "id": self.id,
            "answer": self.answer,
            "raw_response": self._get_combined_response(),
            "question_type": self.question_type,
            "multi_agent_info": {
                "primary_answer": self.primary_answer,
                "verifier_used": self.verifier_used,
                "verifier_method": self.verifier_method if self.verifier_used else None,
                "verifier_answer": self.verifier_answer if self.verifier_used else None,
            }
        }
    
    def _get_combined_response(self) -> str:
        """1차 Agent와 Verifier 응답을 합친 전체 응답"""
        response = f"[Primary Agent]\n{self.primary_raw_response}"
        if self.verifier_used:
            response += f"\n\n[Verifier]\n{self.verifier_raw_response}"
        return response


def parse_answer_from_response(response: str, num_choices: int = 5) -> int:
    """
    LLM 응답에서 정답 숫자 추출 (기존 로직 그대로)
    
    Returns:
        추출된 정답 (1~5), 파싱 불가 시 0
    """
    if not response:
        return 0
    
    response = response.strip()
    
    # 패턴 1: "정답: X" 또는 "정답 X" 형태
    pattern1 = r'정답\s*[:\s]\s*(\d)'
    match1 = re.search(pattern1, response)
    if match1:
        answer = int(match1.group(1))
        if 1 <= answer <= num_choices:
            return answer
    
    # 패턴 2: 응답 끝부분의 숫자
    pattern2 = r'(\d)\s*$'
    match2 = re.search(pattern2, response)
    if match2:
        answer = int(match2.group(1))
        if 1 <= answer <= num_choices:
            return answer
    
    # 패턴 3: 응답 전체에서 마지막으로 나타나는 유효 숫자
    valid_numbers = [str(i) for i in range(1, num_choices + 1)]
    for char in reversed(response):
        if char in valid_numbers:
            return int(char)
    
    return 0


@dataclass
class PrimaryAgentResult:
    """1차 Agent (기존 CoT) 결과"""
    answer: int
    raw_response: str
    confidence: str  # high, medium, low


class MultiAgentProcessor:
    """
    Multi-Agent 문제 풀이 프로세서 (개선 버전)
    
    기존 CoT 방식을 1차 Agent로 사용하고,
    파싱 실패 시에만 Verifier를 호출합니다.
    """
    
    def __init__(
        self, 
        client: AsyncAPIClient,
        system_prompt: str = None,
        use_type_specific_prompt: bool = True,
        use_cot: bool = True,
    ):
        self.client = client
        self.verifier = VerifierAgent(client)
        self.system_prompt = system_prompt
        self.use_type_specific_prompt = use_type_specific_prompt
        self.use_cot = use_cot
    
    async def _run_primary_agent(
        self,
        item: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> PrimaryAgentResult:
        """
        1차 Agent 실행 (기존 CoT 방식)
        """
        async with semaphore:
            # 문제 유형 가져오기
            question_type = item.get('question_type', QuestionType.DEFAULT)
            
            # 프롬프트 생성 (기존 방식 그대로)
            user_message = format_question_message(
                paragraph=item['paragraph'],
                question=item['question'],
                question_plus=item.get('question_plus', ''),
                choices_list=item['choices'],
                question_type=question_type if self.use_type_specific_prompt else QuestionType.DEFAULT,
                use_cot=self.use_cot
            )
            
            # 메시지 구성
            if self.use_type_specific_prompt:
                messages = create_messages(user_message, question_type=question_type)
            else:
                messages = create_messages(user_message, self.system_prompt)
            
            raw_response = ""
            answer = 0
            all_responses = []
            
            # 최대 MAX_RETRY 횟수만큼 시도
            for attempt in range(1, MAX_RETRY + 1):
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
            
            # 확신도 결정 (간단한 휴리스틱)
            confidence = "high" if answer > 0 else "low"
            
            return PrimaryAgentResult(
                answer=answer,
                raw_response=raw_response,
                confidence=confidence
            )
    
    async def process_single_item(
        self,
        item: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> MultiAgentResult:
        """
        단일 문제를 Multi-Agent 방식으로 처리합니다.
        
        Args:
            item: 문제 데이터 (id, paragraph, question, question_plus, choices, ...)
            semaphore: 동시 요청 제한용 세마포어
        
        Returns:
            MultiAgentResult 객체
        """
        # 1. 1차 Agent 실행 (기존 CoT 방식)
        primary_result = await self._run_primary_agent(item, semaphore)
        
        # 2. 검증 필요 여부 확인 (답이 1~5가 아닐 때만)
        num_choices = len(item['choices'])
        needs_verification = (
            primary_result.answer < 1 or 
            primary_result.answer > num_choices
        )
        
        # 3. 필요시 Verifier Agent 호출
        verifier_result = None
        if needs_verification:
            # Verifier용 SolverResult 형태로 변환
            from .solver_agent import SolverResult
            solver_like_result = SolverResult(
                choice_evaluations=[],
                final_answer=primary_result.answer,
                confidence=primary_result.confidence,
                raw_response=primary_result.raw_response,
                needs_verification=True
            )
            
            verifier_result = await self.verifier.verify(
                paragraph=item['paragraph'],
                question=item['question'],
                question_plus=item.get('question_plus', ''),
                choices=item['choices'],
                solver_result=solver_like_result,
                semaphore=semaphore,
            )
        
        # 4. 최종 답 결정
        if verifier_result is not None:
            final_answer = verifier_result.final_answer
        else:
            final_answer = primary_result.answer
        
        # 5. 결과 구성
        question_type = item.get('question_type')
        if hasattr(question_type, 'value'):
            question_type = question_type.value
        elif question_type is None:
            question_type = 'default'
        
        return MultiAgentResult(
            id=item['id'],
            final_answer=final_answer,
            answer=str(final_answer) if final_answer > 0 else "0",
            primary_answer=primary_result.answer,
            primary_raw_response=primary_result.raw_response,
            verifier_used=(verifier_result is not None),
            verifier_method=verifier_result.method_used if verifier_result else "",
            verifier_answer=verifier_result.final_answer if verifier_result else 0,
            verifier_raw_response=verifier_result.raw_response_re_solve + "\n---\n" + verifier_result.raw_response_review if verifier_result else "",
            question_type=question_type,
        )
    
    async def process_batch(
        self,
        items: List[Dict[str, Any]],
        semaphore: asyncio.Semaphore,
    ) -> List[MultiAgentResult]:
        """
        여러 문제를 병렬로 처리합니다.
        
        Args:
            items: 문제 데이터 리스트
            semaphore: 동시 요청 제한용 세마포어
        
        Returns:
            MultiAgentResult 리스트
        """
        tasks = [
            self.process_single_item(item, semaphore)
            for item in items
        ]
        
        results = await asyncio.gather(*tasks)
        return list(results)
