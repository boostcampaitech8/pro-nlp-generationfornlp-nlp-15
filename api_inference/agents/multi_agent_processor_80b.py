"""
Multi-Agent Processor 80B: 80B Verifier를 사용하는 Multi-Agent 처리

Primary Agent (32B CoT)와 Verifier (80B)를 조율하여 문제를 해결합니다.
"""
import asyncio
from typing import Any

from .primary_agent import PrimaryAgent
from .verifier_80b_agent import Verifier80BAgent
from ..prompts import QuestionType


class MultiAgentProcessor80B:
    """
    80B Verifier를 사용하는 Multi-Agent Processor
    
    Primary Agent로 문제를 풀이하고, 필요시 80B Verifier로 검증합니다.
    """
    
    def __init__(
        self,
        primary_agent: PrimaryAgent,
        verifier_80b: Verifier80BAgent,
        verify_all: bool = False,
        verify_threshold: float | None = None,
    ):
        self.primary_agent = primary_agent
        self.verifier_80b = verifier_80b
        self.verify_all = verify_all
        self.verify_threshold = verify_threshold
    
    async def process_single_item(
        self,
        item: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        """
        Multi-Agent 방식으로 단일 문제 처리
        
        Args:
            item: 문제 데이터 딕셔너리
            semaphore: 동시 요청 제한용 세마포어
        
        Returns:
            결과 딕셔너리 {
                "id": str,
                "answer": str,  # "0" ~ "5"
                "raw_response": str,
                "question_type": str,
                "multi_agent_info": dict
            }
        """
        # 1차 Agent: 기존 CoT 방식
        primary_answer, primary_response = await self.primary_agent.solve(item, semaphore)
        
        # 80B Verifier 호출 여부 결정
        num_choices = len(item['choices'])
        needs_verification = False
        
        if self.verify_all:
            # 모든 문제 검증
            needs_verification = True
        elif self.verify_threshold is not None:
            # 확신도 기반 (logit 기반은 추후 구현 가능)
            # 현재는 파싱 실패 시만
            needs_verification = (primary_answer < 1 or primary_answer > num_choices)
        else:
            # 기본: 파싱 실패 시만
            needs_verification = (primary_answer < 1 or primary_answer > num_choices)
        
        # 80B Verifier 호출
        verifier_answer = 0
        verifier_response = ""
        verifier_used = False
        
        if needs_verification:
            verifier_used = True
            verifier_answer, verifier_response = await self.verifier_80b.verify(
                paragraph=item['paragraph'],
                question=item['question'],
                question_plus=item.get('question_plus', ''),
                choices=item['choices'],
                primary_cot=primary_response,
                semaphore=semaphore,
            )
        
        # 최종 답 결정
        if verifier_used and verifier_answer > 0:
            final_answer = verifier_answer
        else:
            final_answer = primary_answer
        
        # 결과 구성
        question_type = item.get('question_type', QuestionType.DEFAULT)
        question_type_str = question_type.value if hasattr(question_type, 'value') else str(question_type)
        
        # raw_response 구성
        raw_response = f"[Primary Agent]\n{primary_response}"
        if verifier_used:
            raw_response += f"\n\n[80B Verifier]\n{verifier_response}"
        
        return {
            "id": item['id'],
            "answer": str(final_answer) if final_answer > 0 else "0",
            "raw_response": raw_response,
            "question_type": question_type_str,
            "multi_agent_info": {
                "primary_answer": primary_answer,
                "verifier_used": verifier_used,
                "verifier_answer": verifier_answer if verifier_used else None,
            }
        }
    
    async def process_batch(
        self,
        items: list[dict[str, Any]],
        semaphore: asyncio.Semaphore,
    ) -> list[dict[str, Any]]:
        """
        여러 문제를 병렬로 처리합니다.
        
        Args:
            items: 문제 데이터 리스트
            semaphore: 동시 요청 제한용 세마포어
        
        Returns:
            결과 딕셔너리 리스트
        """
        tasks = [
            self.process_single_item(item, semaphore)
            for item in items
        ]
        
        results = await asyncio.gather(*tasks)
        return list(results)

