"""
Standalone script to upload an already-generated GGUF to HuggingFace.
Use this to retry an upload without re-running training.

Usage:
    python3 upload_gguf.py

Reads config from config.yaml and HF_TOKEN from .env or environment.
"""
import os
import sys
import yaml
from dotenv import load_dotenv

load_dotenv()

# --- Load config ---
with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

_preset = _cfg["presets"][_cfg["model_preset"]]
_hub = _cfg["hub"]

GGUF_DIR     = _preset["gguf_dir"]
HF_TOKEN     = _hub.get("hf_token") or os.environ.get("HF_TOKEN")
HF_REPO_GGUF = _hub.get("hf_repo_gguf")

# --- Validate ---
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set. Add it to .env or export HF_TOKEN=hf_xxxx")
    sys.exit(1)

if not HF_REPO_GGUF:
    print("ERROR: hub.hf_repo_gguf not set in config.yaml")
    sys.exit(1)

if not os.path.isdir(GGUF_DIR):
    print(f"ERROR: GGUF directory not found: {GGUF_DIR}")
    print("Run the full training script first to generate the GGUF.")
    sys.exit(1)

# --- List files ---
files = sorted(os.scandir(GGUF_DIR), key=lambda e: e.name)
if not files:
    print(f"ERROR: {GGUF_DIR} is empty — GGUF generation may have failed.")
    sys.exit(1)

print(f"Files to upload from {GGUF_DIR}:")
total_bytes = 0
for f in files:
    size_gb = f.stat().st_size / 1e9
    total_bytes += f.stat().st_size
    print(f"  {f.name:<50} {size_gb:.2f} GB")
print(f"  Total: {total_bytes / 1e9:.2f} GB")
print()

# --- Upload ---
from huggingface_hub import HfApi

api = HfApi(token=HF_TOKEN)
api.create_repo(HF_REPO_GGUF, repo_type="model", exist_ok=True)

print(f"Uploading to https://huggingface.co/{HF_REPO_GGUF} ...")
api.upload_folder(
    folder_path=GGUF_DIR,
    repo_id=HF_REPO_GGUF,
    repo_type="model",
)

print()
print(f"Upload complete!")
print(f"  Repo:  https://huggingface.co/{HF_REPO_GGUF}")
print(f"  Ollama: ollama run hf.co/{HF_REPO_GGUF}")
