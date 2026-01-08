"""
Multi-Agent 추론 시스템

Agents:
- VerifierAgent: 답 검증 및 재풀이 (32B/공통)
- Verifier80BAgent: 80B 모델 전용 Verifier
- PrimaryAgent: Primary Agent (32B CoT)
- MultiAgentProcessor: 전체 흐름 조율 (32B Verifier 사용)
- MultiAgentProcessor80B: 전체 흐름 조율 (80B Verifier 사용)
"""

from .verifier_agent import VerifierAgent
from .verifier_80b_agent import Verifier80BAgent
from .primary_agent import PrimaryAgent
from ..utils.answer_parser import parse_answer_from_response
from .multi_agent_processor import MultiAgentProcessor
from .multi_agent_processor_80b import MultiAgentProcessor80B

__all__ = [
    "VerifierAgent",
    "Verifier80BAgent",
    "parse_answer_from_response",
    "PrimaryAgent",
    "MultiAgentProcessor",
    "MultiAgentProcessor80B",
]




