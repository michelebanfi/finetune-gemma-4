#!/usr/bin/env python3
"""
Convert merged safetensors model to GGUF and upload to HuggingFace.

Use when you already have the merged fp16 model in gguf_dir (produced by
train.py) but need to re-run conversion or upload.

Usage:
    python3 export_gguf.py   # always re-converts and overwrites existing HF file

Reads config from config.yaml and HF_TOKEN from .env or environment.

NOTE: llama.cpp's converter requires transformers<=5.3.0 for Gemma-4.
      This script handles the version pin automatically.
"""
import os
import shutil
import sys
import subprocess
import tempfile

from dotenv import load_dotenv
load_dotenv()

from src.config import load_config

cfg = load_config("config.yaml")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_CPP_DIR = os.path.join(SCRIPT_DIR, "llama.cpp")
CONVERTER = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
GGUF_FILE = os.path.join(SCRIPT_DIR, "model.gguf")

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

# ── Ensure converter is available ─────────────────────────────────────────────
if not os.path.exists(CONVERTER):
    print("llama.cpp not found — cloning ...")
    subprocess.run(
        ["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_DIR],
        check=True,
    )
    req_file = os.path.join(LLAMA_CPP_DIR, "requirements", "requirements-convert_hf_to_gguf.txt")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], check=True)
    print()

# ── Convert ───────────────────────────────────────────────────────────────────
if os.path.exists(GGUF_FILE):
    print(f"Removing stale GGUF: {GGUF_FILE}")
    os.remove(GGUF_FILE)

print(f"Converting to GGUF ({cfg.gguf_quantization}) ...")
print(f"  Input:  {cfg.gguf_dir}")
print(f"  Output: {GGUF_FILE}")
print()

print("Pinning transformers==5.3.0 for Gemma-4 GGUF compat ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.3.0"], check=True)

try:
    subprocess.run(
        [
            sys.executable, CONVERTER,
            cfg.gguf_dir,
            "--outfile", GGUF_FILE,
            "--outtype", cfg.gguf_quantization.lower(),
            "--verbose",
        ],
        check=True,
    )
finally:
    print("Restoring transformers==5.5.0 ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.5.0"], check=True)

size_gb = os.path.getsize(GGUF_FILE) / 1e9
print(f"\nGGUF saved: {GGUF_FILE} ({size_gb:.2f} GB)")

print()

# ── Upload ────────────────────────────────────────────────────────────────────
if not cfg.hf_token:
    print("HF_TOKEN not set — skipping upload.")
    print("To upload manually, set HF_TOKEN in .env and re-run.")
    sys.exit(0)

if not cfg.hf_repo_gguf:
    print("hub.hf_repo_gguf not set in config.yaml — skipping upload.")
    sys.exit(0)

from huggingface_hub import HfApi

api = HfApi(token=cfg.hf_token)
api.create_repo(cfg.hf_repo_gguf, repo_type="model", exist_ok=True)

size_gb = os.path.getsize(GGUF_FILE) / 1e9
print(f"Uploading model.gguf ({size_gb:.2f} GB) to https://huggingface.co/{cfg.hf_repo_gguf} ...")
with tempfile.TemporaryDirectory() as tmp:
    shutil.copy2(GGUF_FILE, os.path.join(tmp, "model.gguf"))
    api.upload_large_folder(
        folder_path=tmp,
        repo_id=cfg.hf_repo_gguf,
        repo_type="model",
        allow_patterns=["model.gguf"],
    )

print()
print("Upload complete!")
print(f"  Repo:   https://huggingface.co/{cfg.hf_repo_gguf}")
print(f"  Ollama: ollama run hf.co/{cfg.hf_repo_gguf}")
