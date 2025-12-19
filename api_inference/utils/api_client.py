"""
OpenAI API 호환 클라이언트 모듈
- 일반 채팅 요청 및 Function Call 요청 지원
- 동기/비동기 모두 지원
"""
import json
from typing import Optional, List, Dict, Any
from openai import OpenAI, AsyncOpenAI


class APIClient:
    """OpenAPI 호환 서버용 API 클라이언트 (동기)"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: inference.yaml에서 로드된 설정 딕셔너리
        """
        api_config = config['api']
        self.client = OpenAI(
            base_url=api_config['base_url'],
            api_key=api_config['api_key'],
            timeout=api_config.get('timeout', 120),
            max_retries=api_config.get('max_retries', 3)
        )
        self.model_name = api_config['model_name']
        self.inference_config = config.get('inference', {})
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        일반 채팅 완성 요청
        
        Args:
            messages: 채팅 메시지 리스트
            temperature: 생성 온도 (None이면 config 값 사용)
            max_tokens: 최대 토큰 수 (None이면 config 값 사용)
        
        Returns:
            LLM 응답 텍스트
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.inference_config.get('temperature', 0.0),
            max_tokens=max_tokens or self.inference_config.get('max_tokens', 32)
        )
        return response.choices[0].message.content
    
    def function_call_completion(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "required",
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Function Call 요청
        
        Args:
            messages: 채팅 메시지 리스트
            tools: Function 정의 리스트
            tool_choice: 도구 선택 방식 ("required", "auto", "none")
            temperature: 생성 온도
        
        Returns:
            Function call 결과 (name, arguments)
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature or self.inference_config.get('temperature', 0.0)
        )
        
        tool_call = response.choices[0].message.tool_calls[0]
        return {
            "name": tool_call.function.name,
            "arguments": json.loads(tool_call.function.arguments)
        }


class AsyncAPIClient:
    """OpenAPI 호환 서버용 비동기 API 클라이언트"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: inference.yaml에서 로드된 설정 딕셔너리
        """
        api_config = config['api']
        self.client = AsyncOpenAI(
            base_url=api_config['base_url'],
            api_key=api_config['api_key'],
            timeout=api_config.get('timeout', 120),
            max_retries=api_config.get('max_retries', 3)
        )
        self.model_name = api_config['model_name']
        self.inference_config = config.get('inference', {})
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        비동기 채팅 완성 요청
        
        Args:
            messages: 채팅 메시지 리스트
            temperature: 생성 온도 (None이면 config 값 사용)
            max_tokens: 최대 토큰 수 (None이면 config 값 사용)
        
        Returns:
            LLM 응답 텍스트
        """
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.inference_config.get('temperature', 0.0),
            max_tokens=max_tokens or self.inference_config.get('max_tokens', 32)
        )
        return response.choices[0].message.content
    
    async def function_call_completion(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        비동기 Function Call 요청
        
        Args:
            messages: 채팅 메시지 리스트
            tools: Function 정의 리스트
            tool_choice: 도구 선택 방식 ("auto", "none", 또는 None이면 config 값 사용)
                         vLLM Hermes parser는 "auto" 권장
            temperature: 생성 온도
        
        Returns:
            Function call 결과 (name, arguments) 또는 fallback content
            - tool_call이 있으면: {"name": ..., "arguments": {...}, "from_tool_call": True}
            - tool_call이 없으면: {"content": ..., "from_tool_call": False}
        """
        # tool_choice 결정: 인자 > config > 기본값 "auto"
        effective_tool_choice = tool_choice or self.inference_config.get('tool_choice', 'auto')
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice=effective_tool_choice,
            temperature=temperature or self.inference_config.get('temperature', 0.0),
            max_tokens=self.inference_config.get('max_tokens', 1024)
        )
        
        message = response.choices[0].message
        
        # tool_calls가 있는 경우
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            return {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
                "from_tool_call": True
            }
        
        # tool_calls가 없는 경우 - content 반환 (fallback)
        return {
            "content": message.content,
            "from_tool_call": False
        }


# Function Call을 위한 도구 정의
ANSWER_SELECTION_TOOL = {
    "type": "function",
    "function": {
        "name": "select_answer",
        "description": "수능 문제의 정답 번호를 선택합니다. 지문과 질문을 분석한 후 가장 적절한 답을 선택하세요.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5],
                    "description": "선택한 정답 번호 (1-5)"
                }
            },
            "required": ["answer"]
        }
    }
}
