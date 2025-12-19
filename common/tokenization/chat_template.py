GEMMA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}"
    "{% if system_message is defined %}{{ system_message }}{% endif %}"
    "{% for message in messages %}{% set content = message['content'] %}"
    "{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
)
