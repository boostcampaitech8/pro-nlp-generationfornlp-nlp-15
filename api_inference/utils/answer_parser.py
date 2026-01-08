"""
답 파싱 유틸리티

LLM 응답에서 정답 숫자를 추출하는 공통 함수
"""
import re


def parse_answer_from_response(response: str, num_choices: int = 5) -> int:
    """
    LLM 응답에서 정답 숫자 추출
    
    Args:
        response: LLM 응답 문자열
        num_choices: 선택지 개수 (기본 5)

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


