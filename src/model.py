"""
Model and tokenizer loading (4-bit QLoRA via Unsloth).
"""
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

from src.config import Config


def load_model_and_tokenizer(cfg: Config):
    print(f"Loading {cfg.model_name} in 4-bit precision...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {round(props.total_memory / 1e9, 1)} GB")

    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
        device_map={"": 0},  # Pin to GPU 0; prevents Trainer from re-moving model
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    return model, tokenizer


def attach_lora(model, cfg: Config):
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,      # Text-only scientific QA
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        random_state=cfg.seed,
    )
    model.print_trainable_parameters()
    return model
