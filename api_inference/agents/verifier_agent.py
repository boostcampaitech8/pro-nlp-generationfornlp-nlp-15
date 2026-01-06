"""
Verifier Agent: 답 검증 및 재풀이

Solver Agent가 유효한 답(1~5)을 내지 못했을 때 호출됩니다.
두 가지 방식을 모두 시도하여 최종 답을 결정합니다:
1. 문제 재풀이 (독립적으로 다시 풀기)
2. Solver의 CoT 검토 후 수정
"""
import re
import asyncio
from dataclasses import dataclass

from api_inference.utils.api_client import AsyncAPIClient

@dataclass
class ChoiceEvaluation:
    """선택지 평가 결과"""
    choice_num: int           # 선택지 번호 (1-5)
    choice_text: str          # 선택지 텍스트
    is_correct: bool | None  # 옳음(True), 틀림(False), 불확실(None)
    reasoning: str            # 판단 근거


@dataclass
class SolverResult:
    """Solver Agent 결과"""
    choice_evaluations: list[ChoiceEvaluation]  # 각 선택지 평가
    final_answer: int         # 최종 답 (1-5, 또는 0=실패)
    confidence: str           # 확신도 (high/medium/low)
    raw_response: str         # 원본 응답
    needs_verification: bool  # 검증 필요 여부

@dataclass
class VerifierResult:
    """Verifier Agent 결과"""
    final_answer: int           # 최종 답 (1-5, 또는 0=실패)
    method_used: str            # 사용된 방법 (re_solve / review_cot / both_agree / vote)
    re_solve_answer: int        # 재풀이 답
    review_answer: int          # CoT 검토 후 답
    raw_response_re_solve: str  # 재풀이 원본 응답
    raw_response_review: str    # CoT 검토 원본 응답
    confidence: str             # 확신도


# 재풀이용 프롬프트
VERIFIER_RE_SOLVE_SYSTEM = """당신은 수능/공무원 시험 문제를 푸는 전문가입니다.
문제를 꼼꼼히 읽고 정확한 답을 찾아주세요.
반드시 마지막 줄에 "정답: (숫자)" 형식으로 답을 출력하세요."""

VERIFIER_RE_SOLVE_USER = """지문:
{paragraph}

질문: {question}
{question_plus}

선택지:
{choices}

위 문제를 풀고, 마지막 줄에 "정답: (숫자)" 형식으로 답을 출력하세요."""


# CoT 검토용 프롬프트
VERIFIER_REVIEW_SYSTEM = """당신은 문제 풀이를 검토하는 전문가입니다.
다른 사람의 풀이를 검토하고, 오류가 있다면 수정하여 올바른 답을 제시해주세요.
반드시 마지막 줄에 "정답: (숫자)" 형식으로 답을 출력하세요."""

VERIFIER_REVIEW_USER = """[문제]
지문:
{paragraph}

질문: {question}
{question_plus}

선택지:
{choices}

[이전 풀이]
{previous_solution}

위 풀이를 검토해주세요:
1. 각 선택지 판단이 올바른지 확인하세요.
2. 논리적 오류가 있다면 지적해주세요.
3. 최종 답이 올바른지 판단하세요.

검토 결과를 서술하고, 마지막 줄에 "정답: (숫자)" 형식으로 올바른 답을 출력하세요."""


class VerifierAgent:
    """
    검증 Agent
    
    Solver가 유효한 답을 내지 못했을 때 호출됩니다.
    두 가지 방식을 모두 시도하여 최종 답을 결정합니다.
    """
    
    def __init__(self, client: AsyncAPIClient):
        self.client = client
    
    async def verify(
        self,
        paragraph: str,
        question: str,
        question_plus: str,
        choices: list[str],
        solver_result: SolverResult,
        semaphore: asyncio.Semaphore,
    ) -> VerifierResult:
        """
        Solver의 결과를 검증하고 최종 답을 결정합니다.
        
        두 가지 방식을 병렬로 실행:
        1. 문제 재풀이
        2. Solver의 CoT 검토
        
        Args:
            paragraph: 지문
            question: 질문
            question_plus: 추가 보기
            choices: 선택지 리스트
            solver_result: Solver Agent의 결과
            semaphore: 동시 요청 제한용 세마포어
        
        Returns:
            VerifierResult 객체
        """
        # 두 방식을 병렬로 실행
        re_solve_task = self._re_solve(
            paragraph, question, question_plus, choices, semaphore
        )
        review_task = self._review_cot(
            paragraph, question, question_plus, choices, solver_result, semaphore
        )
        
        (re_solve_answer, re_solve_response), (review_answer, review_response) = await asyncio.gather(
            re_solve_task, review_task
        )
        
        # 최종 답 결정
        final_answer, method_used, confidence = self._decide_final_answer(
            re_solve_answer, review_answer, choices
        )
        
        return VerifierResult(
            final_answer=final_answer,
            method_used=method_used,
            re_solve_answer=re_solve_answer,
            review_answer=review_answer,
            raw_response_re_solve=re_solve_response,
            raw_response_review=review_response,
            confidence=confidence
        )
    
    async def _re_solve(
        self,
        paragraph: str,
        question: str,
        question_plus: str,
        choices: list[str],
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, str]:
        """문제를 처음부터 다시 풉니다."""
        async with semaphore:
            choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            question_plus_text = f"\n<보기>\n{question_plus}" if question_plus and str(question_plus) != 'nan' else ""
            
            user_message = VERIFIER_RE_SOLVE_USER.format(
                paragraph=paragraph,
                question=question,
                question_plus=question_plus_text,
                choices=choices_text
            )
            
            messages = [
                {"role": "system", "content": VERIFIER_RE_SOLVE_SYSTEM},
                {"role": "user", "content": user_message}
            ]
            
            try:
                response = await self.client.chat_completion(messages=messages)
                answer = self._parse_answer(response, len(choices))
                return answer, response
            except Exception as e:
                return 0, f"Error: {str(e)}"
    
    async def _review_cot(
        self,
        paragraph: str,
        question: str,
        question_plus: str,
        choices: list[str],
        solver_result: SolverResult,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, str]:
        """Solver의 CoT를 검토합니다."""
        async with semaphore:
            choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            question_plus_text = f"\n<보기>\n{question_plus}" if question_plus and str(question_plus) != 'nan' else ""
            
            user_message = VERIFIER_REVIEW_USER.format(
                paragraph=paragraph,
                question=question,
                question_plus=question_plus_text,
                choices=choices_text,
                previous_solution=solver_result.raw_response
            )
            
            messages = [
                {"role": "system", "content": VERIFIER_REVIEW_SYSTEM},
                {"role": "user", "content": user_message}
            ]
            
            try:
                response = await self.client.chat_completion(messages=messages)
                answer = self._parse_answer(response, len(choices))
                return answer, response
            except Exception as e:
                return 0, f"Error: {str(e)}"
    
    def _parse_answer(self, response: str, num_choices: int) -> int:
        """응답에서 정답 숫자를 추출합니다."""
        if not response:
            return 0
        
        patterns = [
            r'정답\s*[:\s]*(\d)',
            r'답\s*[:\s]*(\d)',
            r'(\d)\s*번?이?\s*(정답|맞|올바)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                answer = int(match.group(1))
                if 1 <= answer <= num_choices:
                    return answer
        
        # 패턴 매칭 실패 시, 마지막 숫자 찾기
        for char in reversed(response):
            if char.isdigit():
                answer = int(char)
                if 1 <= answer <= num_choices:
                    return answer
        
        return 0
    
    def _decide_final_answer(
        self,
        re_solve_answer: int,
        review_answer: int,
        choices: list[str]
    ) -> tuple[int, str, str]:
        """
        두 방식의 결과를 종합하여 최종 답을 결정합니다.
        
        Returns:
            (final_answer, method_used, confidence)
        """
        num_choices = len(choices)
        
        # 둘 다 유효한 답을 냈고 일치하는 경우
        if (1 <= re_solve_answer <= num_choices and 
            1 <= review_answer <= num_choices and
            re_solve_answer == review_answer):
            return re_solve_answer, "both_agree", "high"
        
        # 둘 다 유효하지만 다른 경우 -> 재풀이 우선 (더 독립적)
        if 1 <= re_solve_answer <= num_choices and 1 <= review_answer <= num_choices:
            # 투표 방식: 여기서는 재풀이를 우선
            return re_solve_answer, "vote_re_solve", "medium"
        
        # 재풀이만 유효한 경우
        if 1 <= re_solve_answer <= num_choices:
            return re_solve_answer, "re_solve", "medium"
        
        # 검토만 유효한 경우
        if 1 <= review_answer <= num_choices:
            return review_answer, "review_cot", "medium"
        
        # 둘 다 실패한 경우
        return 0, "both_failed", "low"



