"""
Multi-Agent Processor: 전체 흐름 조율

Solver Agent와 Verifier Agent를 조율하여
문제를 풀고 필요시 검증합니다.

Flow:
1. Solver Agent가 선택지별 평가 수행
2. 답이 유효하면 (1~5) → 최종 답 반환
3. 답이 유효하지 않으면 → Verifier Agent 호출
4. Verifier의 결과를 최종 답으로 반환
"""
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

from api_inference.utils.api_client import AsyncAPIClient
from .solver_agent import SolverAgent, SolverResult
from .verifier_agent import VerifierAgent, VerifierResult


@dataclass
class MultiAgentResult:
    """Multi-Agent 추론 결과"""
    id: str                     # 문제 ID
    final_answer: int           # 최종 답 (1-5, 또는 0=실패)
    answer: str                 # 최종 답 문자열 (기존 호환용)
    
    # Solver 결과
    solver_answer: int          # Solver의 답
    solver_confidence: str      # Solver 확신도
    solver_raw_response: str    # Solver 원본 응답
    
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
                "solver_answer": self.solver_answer,
                "solver_confidence": self.solver_confidence,
                "verifier_used": self.verifier_used,
                "verifier_method": self.verifier_method if self.verifier_used else None,
                "verifier_answer": self.verifier_answer if self.verifier_used else None,
            }
        }
    
    def _get_combined_response(self) -> str:
        """Solver와 Verifier 응답을 합친 전체 응답"""
        response = f"[Solver]\n{self.solver_raw_response}"
        if self.verifier_used:
            response += f"\n\n[Verifier]\n{self.verifier_raw_response}"
        return response


class MultiAgentProcessor:
    """
    Multi-Agent 문제 풀이 프로세서
    
    Solver와 Verifier를 조율하여 문제를 풉니다.
    """
    
    def __init__(self, client: AsyncAPIClient):
        self.client = client
        self.solver = SolverAgent(client)
        self.verifier = VerifierAgent(client)
    
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
        # 1. Solver Agent 실행
        solver_result = await self.solver.solve(
            paragraph=item['paragraph'],
            question=item['question'],
            question_plus=item.get('question_plus', ''),
            choices=item['choices'],
            semaphore=semaphore,
        )
        
        # 2. 검증 필요 여부 확인 (답이 1~5가 아닐 때만)
        num_choices = len(item['choices'])
        needs_verification = (
            solver_result.final_answer < 1 or 
            solver_result.final_answer > num_choices
        )
        
        # 3. 필요시 Verifier Agent 호출
        verifier_result = None
        if needs_verification:
            verifier_result = await self.verifier.verify(
                paragraph=item['paragraph'],
                question=item['question'],
                question_plus=item.get('question_plus', ''),
                choices=item['choices'],
                solver_result=solver_result,
                semaphore=semaphore,
            )
        
        # 4. 최종 답 결정
        if verifier_result is not None:
            final_answer = verifier_result.final_answer
        else:
            final_answer = solver_result.final_answer
        
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
            solver_answer=solver_result.final_answer,
            solver_confidence=solver_result.confidence,
            solver_raw_response=solver_result.raw_response,
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

