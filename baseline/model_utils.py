import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

# 1. Gemma 전용 채팅 템플릿 상수
GEMMA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}"
    "{% if system_message is defined %}{{ system_message }}{% endif %}"
    "{% for message in messages %}{% set content = message['content'] %}"
    "{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
)

# 2. 프롬프트 양식 상수 (동적 조립을 위해 {content} 사용)
BASE_PROMPT_FORMAT = """지문:
{paragraph}

{question_content}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

def format_question_message(paragraph, question, question_plus, choices_list):
    """
    <보기> 유무를 자동으로 판단하여 최종 프롬프트를 생성합니다.
    """
    # 선택지 문자열 생성
    choices_str = "\n".join([f"{i + 1} - {choice}" for i, choice in enumerate(choices_list)])
    
    # <보기> 유무에 따른 질문 내용 구성
    if question_plus and str(question_plus).strip() and str(question_plus) != 'nan':
        question_content = f"질문:\n{question}\n\n<보기>\n{question_plus}"
    else:
        question_content = f"질문:\n{question}"
    
    return BASE_PROMPT_FORMAT.format(
        paragraph=paragraph,
        question_content=question_content,
        choices=choices_str
    )

def load_model_and_tokenizer(config, is_train=True):
    model_id = config['model']['name_or_path']
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 템플릿 주입 (여기서 한 번만 하면 모든 파일에서 적용됨)
    tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    if is_train:
        model.gradient_checkpointing_enable() 
        model.enable_input_require_grads()
        # 학습 모드: LoRA 설정 적용
        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['lora_alpha'],
            target_modules=config['lora']['target_modules'],
            lora_dropout=config['lora']['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        return model, tokenizer, lora_config
    else:
        # 추론 모드: 학습된 어댑터 로드
        checkpoint_path = f"{config['training']['output_dir']}/final_adapter"
        print(f"Loading Adapter from {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        return model, tokenizer, None