#!/usr/bin/env python3
"""
Gemma-4 Scientific Fine-tuning (Unsloth + QLoRA)
Pipeline: load → LoRA → dataset → train → save adapter → optional GGUF export

Run with:
    python3 train.py              # full run (train + GGUF export)
    python3 train.py --skip-gguf  # train only; run export_gguf.py separately
Config:   config.yaml  |  Credentials: .env (HF_TOKEN)
"""
import os
import argparse
import subprocess
import sys

from dotenv import load_dotenv
load_dotenv()

os.environ["TQDM_DISABLE"] = "0"

# ── Bootstrap dependencies ───────────────────────────────────────────────────
def _install():
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "torch>=2.8.0", "triton>=3.4.0",
        "torchvision", "bitsandbytes",
        "unsloth", "unsloth_zoo>=2026.4.6",
        "transformers==5.5.0", "torchcodec", "timm",
    ], check=True)

_install()

# ── Imports (after install) ───────────────────────────────────────────────────
from src.config import load_config
from src.model import load_model_and_tokenizer, attach_lora
from src.data import build_training_dataset
from src.training import build_trainer, run_training
from src.gguf import merge_and_save, convert_to_gguf, upload_to_hub
from src.evaluation import run_comparison

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--skip-gguf", action="store_true",
                    help="Skip GGUF export — run export_gguf.py separately")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
cfg = load_config("config.yaml", skip_gguf=args.skip_gguf)
print(f"Active preset: {cfg.model_name}")

# ── Model + LoRA ──────────────────────────────────────────────────────────────
model, tokenizer = load_model_and_tokenizer(cfg)
model = attach_lora(model, cfg)

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = build_training_dataset(cfg, tokenizer)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = build_trainer(model, tokenizer, dataset, cfg)
run_training(trainer)

# ── Save LoRA adapter ─────────────────────────────────────────────────────────
print(f"\nSaving LoRA adapter to {cfg.output_dir}...")
model.save_pretrained(cfg.output_dir)
tokenizer.save_pretrained(cfg.output_dir)
print("Adapter saved.")

# ── GGUF export ───────────────────────────────────────────────────────────────
if cfg.export_gguf:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    merge_and_save(model, tokenizer, cfg)
    gguf_file = convert_to_gguf(cfg, script_dir)
    upload_to_hub(gguf_file, cfg)

# ── Qualitative evaluation ────────────────────────────────────────────────────
run_comparison(model, tokenizer, cfg)
