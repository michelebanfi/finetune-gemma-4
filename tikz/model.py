"""
Model and tokenizer loading for TikZ task (4-bit QLoRA via Unsloth).
Vision layers are finetuned to enable TikZ diagram understanding.
"""
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

from tikz.config import TikZConfig


def load_model_and_tokenizer(cfg: TikZConfig):
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
        device_map={"": 0},  # Pin to GPU 0
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    return model, tokenizer


def attach_lora(model, cfg: TikZConfig):
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.finetune_vision_layers,   # True for TikZ (vs False for sci)
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
