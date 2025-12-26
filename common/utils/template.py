from transformers import PreTrainedTokenizerBase
import logging

logger = logging.getLogger(__name__)


def get_response_template(tokenizer: PreTrainedTokenizerBase) -> str:
    """
    토크나이저의 chat_template을 분석하여, Assistant의 응답이 시작되는
    response_template 문자열을 자동으로 추출합니다.

    작동 원리:
    1. (System + User) 메시지를 포맷팅한 결과(A)를 구합니다.
    2. (System + User + Assistant) 메시지를 포맷팅한 결과(B)를 구합니다.
    3. B에서 A를 뺀 부분의 앞부분을 분석하여 템플릿을 찾습니다.
    (간단하게는 B에서 A 문자열을 제거하면 남는 부분이 응답인데, 그 앞부분이 템플릿입니다.)

    하지만 토크나이저마다(Llama3, ChatML 등) 동작이 다르므로 가장 안전한 방법은
    add_generation_prompt=True를 활용하는 것입니다.
    """

    # 1. 템플릿이 없는 경우 예외 처리
    if not tokenizer.chat_template:
        logger.warning("No chat_template found. Defaulting to ChatML format.")
        return "<|im_start|><|assistant|>"

    dummy_messages = [
        {"role": "user", "content": "FOR_TEMPLATE_DETECTION"},
    ]

    try:
        # prompt_text: 유저 턴이 끝난 직후까지의 텍스트 (generation prompt 포함)
        # 예: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
        prompt_text = tokenizer.apply_chat_template(
            dummy_messages, tokenize=False, add_generation_prompt=True
        )

        # no_gen_text: generation prompt 없는 텍스트
        # 예: <|im_start|>user\n...<|im_end|>\n
        no_gen_text = tokenizer.apply_chat_template(
            dummy_messages, tokenize=False, add_generation_prompt=False
        )

        # prompt_text의 끝부분이 response_template일 가능성이 높음
        if prompt_text.startswith(no_gen_text):
            response_template = prompt_text[len(no_gen_text) :]

            # 혹시라도 비어있거나 너무 길다면 안전장치
            if not response_template or len(response_template) > 50:
                # Fallback: ChatML/Gemma 등 흔한 패턴 확인
                if "<|im_start|>" in prompt_text:
                    return "<|im_start|><|assistant|>"
                if "<start_of_turn>" in prompt_text:
                    return "<start_of_turn>model\n"

            return response_template

        # 위 방식으로 안 구해지면 단순 하드코딩 Fallback
        if "<|im_start|>" in prompt_text:
            return "<|im_start|><|assistant|>"

        return "\n### Response:\n"  # Alpaca style fallback

    except Exception as e:
        logger.warning(f"Failed to auto-detect template: {e}. Using fallback.")
        return "<|im_start|><|assistant|>"
