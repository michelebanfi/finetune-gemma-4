#!/usr/bin/env python3
"""
Convert merged safetensors model to GGUF and upload to HuggingFace.

Use when you already have the merged fp16 model in gguf_dir (produced by
train.py) but need to re-run conversion or upload.

Usage:
    python3 export_gguf.py   # always re-converts and overwrites existing HF file

Reads config from sci/config.yaml and HF_TOKEN from .env or environment.

NOTE: llama.cpp's converter requires transformers<=5.3.0 for Gemma-4.
      This script handles the version pin automatically.
"""
import os
import argparse
import sys

from dotenv import load_dotenv
load_dotenv()

from sci.config import load_config
from common.gguf import convert_to_gguf, upload_to_hub

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="sci/config.yaml",
                    help="Path to scientific config YAML (default: sci/config.yaml)")
args = parser.parse_args()

cfg = load_config(args.config)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Validate ──────────────────────────────────────────────────────────────────
if not os.path.isdir(cfg.gguf_dir):
    print(f"ERROR: merged model directory not found: {cfg.gguf_dir}")
    print("Run train.py first to produce the merged fp16 model.")
    sys.exit(1)

merged_files = [f for f in os.listdir(cfg.gguf_dir) if f.endswith(".safetensors")]
if not merged_files:
    print(f"ERROR: No .safetensors files found in {cfg.gguf_dir}")
    print("The model merge step may have failed.")
    sys.exit(1)

print(f"Merged model: {cfg.gguf_dir}")
for f in sorted(os.listdir(cfg.gguf_dir)):
    size_gb = os.path.getsize(os.path.join(cfg.gguf_dir, f)) / 1e9
    print(f"  {f:<50} {size_gb:.2f} GB")
print()

gguf_file = convert_to_gguf(cfg, SCRIPT_DIR)

print()

# ── Upload ────────────────────────────────────────────────────────────────────
if not cfg.hf_token:
    print("HF_TOKEN not set — skipping upload.")
    print("To upload manually, set HF_TOKEN in .env and re-run.")
    sys.exit(0)

if not cfg.hf_repo_gguf:
    print("hub.hf_repo_gguf not set in sci/config.yaml — skipping upload.")
    sys.exit(0)

readme = os.path.join(SCRIPT_DIR, "sci", "README.md")
upload_to_hub(gguf_file, cfg, readme_path=readme)
