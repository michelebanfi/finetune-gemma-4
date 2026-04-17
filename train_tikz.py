#!/usr/bin/env python3
"""
Gemma-4 TikZ Fine-tuning (Unsloth + QLoRA)
Pipeline: load → LoRA (vision+language) → dataset → train → save adapter → optional GGUF export

Run with:
    python3 train_tikz.py              # full run (train + GGUF export)
    python3 train_tikz.py --skip-gguf  # train only; run export_gguf.py separately
    python3 train_tikz.py --smoke-test # 100 examples, quick pipeline check
Config:   tikz/config.yaml  |  Credentials: .env (HF_TOKEN)
"""
import os
import argparse

from dotenv import load_dotenv
from common.bootstrap import install_train_dependencies
from common.gguf import merge_and_save, convert_to_gguf, upload_to_hub

load_dotenv()

os.environ["TQDM_DISABLE"] = "0"

# ── Bootstrap dependencies ───────────────────────────────────────────────────
install_train_dependencies()

# ── Imports (after install) ───────────────────────────────────────────────────
from tikz.config import load_tikz_config
from tikz.model import load_model_and_tokenizer, attach_lora
from tikz.data import build_tikz_training_dataset
from tikz.training import build_trainer, run_training

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--skip-gguf", action="store_true",
                    help="Skip GGUF export — run export_gguf.py separately")
parser.add_argument("--smoke-test", action="store_true",
                    help="Use only 100 training examples to verify the pipeline")
parser.add_argument("--config", default="tikz/config.yaml",
                    help="Path to TikZ config YAML (default: tikz/config.yaml)")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
cfg = load_tikz_config(args.config, skip_gguf=args.skip_gguf)

if args.smoke_test:
    cfg.subset_size = 100
    cfg.max_steps = 20
    print("*** SMOKE TEST MODE: 100 examples, 20 steps ***")

print(f"Active preset:        {cfg.model_name}")
print(f"Max seq length:       {cfg.max_seq_length}")
print(f"Vision layers:        {cfg.finetune_vision_layers}")
print(f"Dataset:              {cfg.dataset_name} ({cfg.subset_size:,} rows)")
print(f"Task ratios:          {cfg.task_ratios}")

# ── Model + LoRA ──────────────────────────────────────────────────────────────
model, tokenizer = load_model_and_tokenizer(cfg)
model = attach_lora(model, cfg)

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = build_tikz_training_dataset(cfg, tokenizer)

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
    readme = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tikz", "README.md")
    upload_to_hub(gguf_file, cfg, readme_path=readme)
