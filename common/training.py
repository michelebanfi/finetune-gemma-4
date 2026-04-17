"""Shared SFTTrainer build/run helpers."""
import torch
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only


def build_trainer(model, tokenizer, dataset, cfg) -> SFTTrainer:
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    print(f"Mixed precision: {'bf16' if use_bf16 else 'fp16'}")

    gpu_stats = torch.cuda.get_device_properties(0)
    start_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    max_memory = round(gpu_stats.total_memory / 1024**3, 2)
    print(f"GPU 0: {gpu_stats.name} | VRAM: {max_memory} GB total, {start_memory} GB reserved")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation,
            gradient_checkpointing=True,
            warmup_steps=cfg.warmup_steps,
            num_train_epochs=cfg.num_train_epochs,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=1,
            save_strategy="steps",
            save_steps=500,
            output_dir=cfg.output_dir,
            optim="adamw_8bit",
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=cfg.lr_scheduler,
            seed=cfg.seed,
            report_to="none",
            dataset_num_proc=4,
            packing=False,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    print("Verifying response masking (should see only model turns):")
    decoded_labels = tokenizer.decode(
        [tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]
    ).replace(tokenizer.pad_token, " ")
    print(decoded_labels[:300])
    print()

    return trainer


def run_training(trainer: SFTTrainer, training_label: str = "fine-tuning") -> dict:
    print("=" * 60)
    print(f"Starting {training_label}...")
    print("=" * 60)

    trainer_stats = trainer.train()

    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory = round(gpu_stats.total_memory / 1024**3, 2)
    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 2)

    print(f"\nTraining complete in {round(trainer_stats.metrics['train_runtime'] / 60, 1)} minutes")
    print(f"Peak VRAM: {used_memory} GB / {max_memory} GB ({round(used_memory / max_memory * 100, 1)}%)")

    return trainer_stats.metrics
