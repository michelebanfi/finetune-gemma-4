"""Scientific sub-project training helpers."""

from common import training as common_training
from sci.config import Config


def build_trainer(model, tokenizer, dataset, cfg: Config):
    return common_training.build_trainer(model, tokenizer, dataset, cfg)


def run_training(trainer) -> dict:
    return common_training.run_training(trainer, training_label="scientific fine-tuning")
