"""Load and expose training configuration for the scientific sub-project."""
import os
from dataclasses import dataclass
import yaml


@dataclass
class Config:
    # Model
    model_name: str
    max_seq_length: int

    # LoRA
    lora_r: int
    lora_alpha: int
    lora_dropout: float

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
    os_data_subset: int
    sciriff_data_subset: int

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


def _default_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path: str | None = None, skip_gguf: bool = False) -> Config:
    path = path or _default_config_path()
    with open(path) as f:
        raw = yaml.safe_load(f)

    preset = raw["presets"][raw["model_preset"]]
    train = raw["training"]
    ds = raw["dataset"]
    export = raw["export"]
    hub = raw["hub"]

    return Config(
        model_name=preset["model_name"],
        max_seq_length=preset["max_seq_length"],
        lora_r=preset["lora_r"],
        lora_alpha=preset["lora_alpha"],
        lora_dropout=train["lora_dropout"],
        per_device_batch_size=preset["per_device_batch_size"],
        gradient_accumulation=preset["gradient_accumulation"],
        learning_rate=train["learning_rate"],
        warmup_steps=train["warmup_steps"],
        lr_scheduler=train["lr_scheduler"],
        num_train_epochs=train["num_train_epochs"],
        max_steps=train["max_steps"],
        weight_decay=train["weight_decay"],
        seed=train["seed"],
        os_data_subset=ds["os_data_subset"],
        sciriff_data_subset=ds["sciriff_data_subset"],
        output_dir=preset["output_dir"],
        gguf_dir=preset["gguf_dir"],
        export_gguf=export["gguf"] and not skip_gguf,
        gguf_quantization=export["gguf_quantization"],
        hf_token=hub["hf_token"] or os.environ.get("HF_TOKEN"),
        hf_repo=hub["hf_repo"],
        hf_repo_gguf=hub["hf_repo_gguf"],
    )
