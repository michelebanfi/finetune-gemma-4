"""Shared model loading and LoRA attachment helpers."""
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template


def load_model_and_tokenizer(model_name: str, max_seq_length: int):
    print(f"Loading {model_name} in 4-bit precision...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {round(props.total_memory / 1e9, 1)} GB")

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        device_map={"": 0},
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    return model, tokenizer


def attach_lora(
    model,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    seed: int,
    finetune_vision_layers: bool,
):
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=finetune_vision_layers,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        random_state=seed,
    )
    model.print_trainable_parameters()
    return model
