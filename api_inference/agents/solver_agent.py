"""
Solver Agent: 선택지별 독립 평가 후 답 결정

한 번의 LLM 호출에서 모든 선택지를 평가하고,
구조화된 출력으로 각 선택지의 옳고 그름을 판단합니다.
"""
import re
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from api_inference.utils.api_client import AsyncAPIClient


@dataclass
class ChoiceEvaluation:
    """선택지 평가 결과"""
    choice_num: int           # 선택지 번호 (1-5)
    choice_text: str          # 선택지 텍스트
    is_correct: Optional[bool]  # 옳음(True), 틀림(False), 불확실(None)
    reasoning: str            # 판단 근거


@dataclass
class SolverResult:
    """Solver Agent 결과"""
    choice_evaluations: List[ChoiceEvaluation]  # 각 선택지 평가
    final_answer: int         # 최종 답 (1-5, 또는 0=실패)
    confidence: str           # 확신도 (high/medium/low)
    raw_response: str         # 원본 응답
    needs_verification: bool  # 검증 필요 여부


# Solver Agent 전용 프롬프트
SOLVER_SYSTEM_PROMPT = """당신은 수능/공무원 시험 문제를 분석하는 전문가입니다.
각 선택지를 독립적으로 평가하고, 구조화된 형식으로 응답해주세요.

응답 형식을 반드시 지켜주세요:
[선택지 평가]
1번: [O/X/△] - (판단 근거)
2번: [O/X/△] - (판단 근거)
3번: [O/X/△] - (판단 근거)
4번: [O/X/△] - (판단 근거)
5번: [O/X/△] - (판단 근거)

[최종 답]
정답: (숫자)
확신도: (높음/보통/낮음)

O = 옳은 선택지, X = 틀린 선택지, △ = 불확실"""


SOLVER_USER_PROMPT = """지문:
{paragraph}

질문: {question}
{question_plus}

선택지:
{choices}

위 문제의 각 선택지가 옳은지 틀린지 독립적으로 평가해주세요.
그리고 질문의 요구사항(옳은 것/옳지 않은 것)에 맞는 최종 답을 선택해주세요.

반드시 아래 형식으로 응답해주세요:
[선택지 평가]
1번: [O/X/△] - (판단 근거)
2번: [O/X/△] - (판단 근거)
...

[최종 답]
정답: (숫자)
확신도: (높음/보통/낮음)"""


class SolverAgent:
    """
    선택지별 독립 평가 Agent
    
    각 선택지를 O/X/△로 평가하고 최종 답을 결정합니다.
    """
    
    def __init__(self, client: AsyncAPIClient):
        self.client = client
    
    async def solve(
        self,
        paragraph: str,
        question: str,
        question_plus: str,
        choices: List[str],
        semaphore: asyncio.Semaphore,
    ) -> SolverResult:
        """
        문제를 풀고 선택지별 평가 결과를 반환합니다.
        
        Args:
            paragraph: 지문
            question: 질문
            question_plus: 추가 보기 (ㄱ, ㄴ, ㄷ 등)
            choices: 선택지 리스트
            semaphore: 동시 요청 제한용 세마포어
        
        Returns:
            SolverResult 객체
        """
        async with semaphore:
            # 프롬프트 구성
            choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            question_plus_text = f"\n<보기>\n{question_plus}" if question_plus and str(question_plus) != 'nan' else ""
            
            user_message = SOLVER_USER_PROMPT.format(
                paragraph=paragraph,
                question=question,
                question_plus=question_plus_text,
                choices=choices_text
            )
            
            messages = [
                {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
            
            try:
                response = await self.client.chat_completion(messages=messages)
                return self._parse_response(response, choices)
            except Exception as e:
                # 오류 발생 시 빈 결과 반환
                return SolverResult(
                    choice_evaluations=[],
                    final_answer=0,
                    confidence="low",
                    raw_response=f"Error: {str(e)}",
                    needs_verification=True
                )
    
    def _parse_response(self, response: str, choices: List[str]) -> SolverResult:
        """
        LLM 응답을 파싱하여 SolverResult 객체로 변환합니다.
        """
        if not response:
            return SolverResult(
                choice_evaluations=[],
                final_answer=0,
                confidence="low",
                raw_response="",
                needs_verification=True
            )
        
        evaluations = []
        final_answer = 0
        confidence = "low"
        
        # 선택지 평가 파싱
        for i, choice in enumerate(choices):
            choice_num = i + 1
            # 패턴: "1번: [O/X/△]" 또는 "1번: O" 또는 "① O"
            patterns = [
                rf'{choice_num}번\s*[:\s]*\[?([OX△○×])\]?',
                rf'[①②③④⑤][^OX△○×]*([OX△○×])',
            ]
            
            is_correct = None
            reasoning = ""
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    mark = match.group(1).upper()
                    if mark in ['O', '○']:
                        is_correct = True
                    elif mark in ['X', '×']:
                        is_correct = False
                    else:  # △
                        is_correct = None
                    
                    # 판단 근거 추출 (하이픈 뒤의 내용)
                    reason_match = re.search(rf'{choice_num}번[^-]*-\s*(.+?)(?=\n|$)', response)
                    if reason_match:
                        reasoning = reason_match.group(1).strip()
                    break
            
            evaluations.append(ChoiceEvaluation(
                choice_num=choice_num,
                choice_text=choice,
                is_correct=is_correct,
                reasoning=reasoning
            ))
        
        # 최종 답 파싱
        answer_patterns = [
            r'정답\s*[:\s]*(\d)',
            r'최종\s*답\s*[:\s]*(\d)',
            r'답\s*[:\s]*(\d)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response)
            if match:
                answer = int(match.group(1))
                if 1 <= answer <= len(choices):
                    final_answer = answer
                    break
        
        # 확신도 파싱
        if '높음' in response or 'high' in response.lower():
            confidence = "high"
        elif '보통' in response or 'medium' in response.lower():
            confidence = "medium"
        else:
            confidence = "low"
        
        # 검증 필요 여부 결정
        needs_verification = (
            final_answer == 0 or  # 답을 못 찾음
            final_answer < 1 or final_answer > len(choices) or  # 유효하지 않은 답
            confidence == "low"  # 확신도 낮음 (나중에 옵션으로 활성화 가능)
        )
        
        # 현재 설정: 답이 1~5가 아닐 때만 검증 필요
        needs_verification = (final_answer < 1 or final_answer > len(choices))
        
        return SolverResult(
            choice_evaluations=evaluations,
            final_answer=final_answer,
            confidence=confidence,
            raw_response=response,
            needs_verification=needs_verification
        )



