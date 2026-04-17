"""TikZ sub-project training helpers."""

from common import training as common_training
from tikz.config import TikZConfig


def build_trainer(model, tokenizer, dataset, cfg: TikZConfig):
    return common_training.build_trainer(model, tokenizer, dataset, cfg)


def run_training(trainer) -> dict:
    return common_training.run_training(trainer, training_label="TikZ fine-tuning")
