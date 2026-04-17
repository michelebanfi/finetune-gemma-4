#!/usr/bin/env python3
"""
Gemma-4 Scientific Fine-tuning (Unsloth + QLoRA)
Pipeline: load → LoRA → dataset → train → save adapter → optional GGUF export

Run with:
    python3 train.py              # full run (train + GGUF export)
    python3 train.py --skip-gguf  # train only; run export_gguf.py separately
Config:   sci/config.yaml  |  Credentials: .env (HF_TOKEN)
"""
import os
import argparse

from dotenv import load_dotenv
from common.bootstrap import install_train_dependencies

load_dotenv()

os.environ["TQDM_DISABLE"] = "0"

# ── Bootstrap dependencies ───────────────────────────────────────────────────
install_train_dependencies()

# ── Imports (after install) ───────────────────────────────────────────────────
from sci.config import load_config
from sci.model import load_model_and_tokenizer, attach_lora
from sci.data import build_training_dataset
from sci.training import build_trainer, run_training
from common.gguf import merge_and_save, convert_to_gguf, upload_to_hub
from sci.evaluation import run_comparison

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--skip-gguf", action="store_true",
                    help="Skip GGUF export — run export_gguf.py separately")
parser.add_argument("--config", default="sci/config.yaml",
                    help="Path to scientific config YAML (default: sci/config.yaml)")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
cfg = load_config(args.config, skip_gguf=args.skip_gguf)
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
    readme = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sci", "README.md")
    upload_to_hub(gguf_file, cfg, readme_path=readme)

# ── Qualitative evaluation ────────────────────────────────────────────────────
run_comparison(model, tokenizer, cfg)
