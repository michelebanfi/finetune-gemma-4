"""
Load and expose training configuration from tikz_config.yaml.
"""
import os
from dataclasses import dataclass, field
from typing import Dict
import yaml


@dataclass
class TikZConfig:
    # Model
    model_name: str
    max_seq_length: int

    # LoRA
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    finetune_vision_layers: bool

    # Training
    per_device_batch_size: int
    gradient_accumulation: int
    learning_rate: float
    warmup_steps: int
    lr_scheduler: str
    num_train_epochs: int
    max_steps: int
    weight_decay: float
    seed: int

    # Dataset
    dataset_name: str
    subset_size: int
    eval_dataset: str
    task_ratios: Dict[str, float]

    # Output
    output_dir: str
    gguf_dir: str

    # Export
    export_gguf: bool
    gguf_quantization: str

    # Hub
    hf_token: str | None
    hf_repo: str | None
    hf_repo_gguf: str | None


def load_tikz_config(path: str = "tikz_config.yaml", skip_gguf: bool = False) -> TikZConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    preset = raw["presets"][raw["model_preset"]]
    train = raw["training"]
    ds = raw["dataset"]
    export = raw["export"]
    hub = raw["hub"]

    task_ratios = ds["task_ratios"]
    total = sum(task_ratios.values())
    assert abs(total - 1.0) < 1e-6, f"task_ratios must sum to 1.0, got {total}"

    return TikZConfig(
        model_name=preset["model_name"],
        max_seq_length=preset["max_seq_length"],
        lora_r=preset["lora_r"],
        lora_alpha=preset["lora_alpha"],
        lora_dropout=train["lora_dropout"],
        finetune_vision_layers=train["finetune_vision_layers"],
        per_device_batch_size=preset["per_device_batch_size"],
        gradient_accumulation=preset["gradient_accumulation"],
        learning_rate=train["learning_rate"],
        warmup_steps=train["warmup_steps"],
        lr_scheduler=train["lr_scheduler"],
        num_train_epochs=train["num_train_epochs"],
        max_steps=train["max_steps"],
        weight_decay=train["weight_decay"],
        seed=train["seed"],
        dataset_name=ds["name"],
        subset_size=ds["subset_size"],
        eval_dataset=ds["eval_dataset"],
        task_ratios=task_ratios,
        output_dir=preset["output_dir"],
        gguf_dir=preset["gguf_dir"],
        export_gguf=export["gguf"] and not skip_gguf,
        gguf_quantization=export["gguf_quantization"],
        hf_token=hub["hf_token"] or os.environ.get("HF_TOKEN"),
        hf_repo=hub["hf_repo"],
        hf_repo_gguf=hub["hf_repo_gguf"],
    )
