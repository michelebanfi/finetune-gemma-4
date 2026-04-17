"""Scientific sub-project model helpers."""

from common import modeling
from sci.config import Config


def load_model_and_tokenizer(cfg: Config):
    return modeling.load_model_and_tokenizer(cfg.model_name, cfg.max_seq_length)


def attach_lora(model, cfg: Config):
    return modeling.attach_lora(
        model,
        finetune_vision_layers=False,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        seed=cfg.seed,
    )
