# src/trainer.py
import os
import torch
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM # 추가됨

def run_training(model, tokenizer, train_dataset, val_dataset, config, metrics_obj=None, peft_config=None):
    
    # 1. 토크나이저 설정 (User provided settings)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 2. WandB 설정
    wandb_conf = config.get('wandb', {})
    if wandb_conf:
        if "WANDB_PROJECT" not in os.environ and wandb_conf.get('project'):
            os.environ["WANDB_PROJECT"] = wandb_conf['project']
        if "WANDB_ENTITY" not in os.environ and wandb_conf.get('entity'):
            os.environ["WANDB_ENTITY"] = wandb_conf['entity']
        if wandb_conf.get('name'):
             os.environ["WANDB_NAME"] = wandb_conf['name']

    train_conf = config['training']
    data_conf = config['data']

    # 3. Data Collator 설정 (CompletionOnlyLM 적용)
    # 질문(Prompt) 부분은 마스킹하고, 답변(Response) 부분만 학습합니다.
    response_template = "<start_of_turn>model" # Gemma 모델 기준
    
    print(f"Using DataCollatorForCompletionOnlyLM with template: '{response_template}'")
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # 4. SFTConfig 설정
    sft_config = SFTConfig(
        output_dir=train_conf.get('output_dir', './outputs'),
        do_train=True,
        do_eval=True,
        # 학습 하이퍼파라미터
        num_train_epochs=train_conf.get('num_train_epochs', 3),
        learning_rate=float(train_conf.get('learning_rate', 2.0e-5)),
        per_device_train_batch_size=train_conf.get('per_device_train_batch_size', 1),
        per_device_eval_batch_size=train_conf.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=train_conf.get('gradient_accumulation_steps', 4),
        # 스케줄러 및 최적화
        lr_scheduler_type=train_conf.get('lr_scheduler_type', 'cosine'),
        weight_decay=train_conf.get('weight_decay', 0.01),
        # 로깅 및 저장 전략 (YAML 값 우선, 없으면 에폭 단위)
        logging_steps=train_conf.get('logging_steps', 1),
        save_strategy=train_conf.get('save_strategy', 'epoch'),
        evaluation_strategy=train_conf.get('evaluation_strategy', 'epoch'),
        save_total_limit=train_conf.get('save_total_limit', 2),
        save_only_model=True,
        bf16=train_conf.get('bf16', True), 
        #tf32=train_conf.get('tf32', True),
        gradient_checkpointing=train_conf.get('gradient_checkpointing', True),
        max_seq_length=data_conf.get('max_seq_length', 1024),
        dataset_kwargs={"skip_prepare_dataset": True}, 
        report_to="wandb" if train_conf.get('report_to') != "none" else "none",
        run_name=wandb_conf.get('name', 'gemma-train-run'),
    )

    # 5. SFTTrainer 초기화
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator, # CompletionOnlyLM Collator
        tokenizer=tokenizer,
        
        # Metrics 클래스에서 메서드 연결
        compute_metrics=metrics_obj.compute_metrics if metrics_obj else None,
        preprocess_logits_for_metrics=metrics_obj.preprocess_logits_for_metrics if metrics_obj else None,
        
        peft_config=peft_config,
        args=sft_config,
    )

    print("Starting training...")
    trainer.train()
    
    final_save_path = f"{train_conf['output_dir']}/final_adapter"
    trainer.model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Training finished. Saved to {final_save_path}")