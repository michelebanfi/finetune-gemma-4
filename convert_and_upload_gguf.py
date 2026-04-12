"""
Convert merged safetensors model to GGUF and upload to HuggingFace.

Use this script when you already have the merged fp16 model in GGUF_DIR
(produced by the training script) but need to re-run conversion or upload.

Usage:
    python3 convert_and_upload_gguf.py

Reads config from config.yaml and HF_TOKEN from .env or environment.

NOTE: llama.cpp's converter requires transformers<=5.3.0 for Gemma-4.
      If you see a Gemma4Config import error, run:
          pip install transformers==5.3.0
      then re-run this script. Restore afterwards with:
          pip install transformers==5.5.0
"""
import os
import sys
import subprocess
import yaml
from dotenv import load_dotenv

load_dotenv()

# --- Load config ---
with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

_preset  = _cfg["presets"][_cfg["model_preset"]]
_hub     = _cfg["hub"]
_export  = _cfg["export"]

GGUF_DIR          = _preset["gguf_dir"]
GGUF_QUANTIZATION = _export["gguf_quantization"]
HF_TOKEN          = _hub.get("hf_token") or os.environ.get("HF_TOKEN")
HF_REPO_GGUF      = _hub.get("hf_repo_gguf")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
LLAMA_CPP_DIR = os.path.join(SCRIPT_DIR, "llama.cpp")
CONVERTER     = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
GGUF_FILE     = os.path.join(SCRIPT_DIR, "model.gguf")

# --- Validate inputs ---
if not os.path.isdir(GGUF_DIR):
    print(f"ERROR: merged model directory not found: {GGUF_DIR}")
    print("Run the training script first to produce the merged fp16 model.")
    sys.exit(1)

merged_files = [f for f in os.listdir(GGUF_DIR) if f.endswith(".safetensors")]
if not merged_files:
    print(f"ERROR: No .safetensors files found in {GGUF_DIR}")
    print("The model merge step may have failed.")
    sys.exit(1)

print(f"Merged model: {GGUF_DIR}")
for f in sorted(os.listdir(GGUF_DIR)):
    size_gb = os.path.getsize(os.path.join(GGUF_DIR, f)) / 1e9
    print(f"  {f:<50} {size_gb:.2f} GB")
print()

# --- Ensure llama.cpp converter is available ---
if not os.path.exists(CONVERTER):
    print("llama.cpp not found — cloning ...")
    subprocess.run(
        ["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_DIR],
        check=True,
    )
    req_file = os.path.join(LLAMA_CPP_DIR, "requirements", "requirements-convert_hf_to_gguf.txt")
    subprocess.run(["pip", "install", "-r", req_file], check=True)
    print()

# --- Convert to GGUF ---
if os.path.exists(GGUF_FILE):
    size_gb = os.path.getsize(GGUF_FILE) / 1e9
    print(f"GGUF already exists: {GGUF_FILE} ({size_gb:.2f} GB) — skipping conversion.")
    print("Delete model.gguf to force re-conversion.")
else:
    print(f"Converting to GGUF ({GGUF_QUANTIZATION}) ...")
    print(f"  Input:  {GGUF_DIR}")
    print(f"  Output: {GGUF_FILE}")
    print()
    result = subprocess.run(
        [
            "python3", CONVERTER,
            GGUF_DIR,
            "--outfile", GGUF_FILE,
            "--outtype", GGUF_QUANTIZATION.lower(),
            "--verbose",
        ],
        check=True,
    )
    size_gb = os.path.getsize(GGUF_FILE) / 1e9
    print(f"\nGGUF saved: {GGUF_FILE} ({size_gb:.2f} GB)")

print()

# --- Upload ---
if not HF_TOKEN:
    print("HF_TOKEN not set — skipping upload.")
    print("To upload manually, set HF_TOKEN in .env and re-run.")
    sys.exit(0)

if not HF_REPO_GGUF:
    print("hub.hf_repo_gguf not set in config.yaml — skipping upload.")
    sys.exit(0)

from huggingface_hub import HfApi

api = HfApi(token=HF_TOKEN)
api.create_repo(HF_REPO_GGUF, repo_type="model", exist_ok=True)

size_gb = os.path.getsize(GGUF_FILE) / 1e9
print(f"Uploading model.gguf ({size_gb:.2f} GB) to https://huggingface.co/{HF_REPO_GGUF} ...")
api.upload_file(
    path_or_fileobj=GGUF_FILE,
    path_in_repo="model.gguf",
    repo_id=HF_REPO_GGUF,
    repo_type="model",
)

print()
print("Upload complete!")
print(f"  Repo:   https://huggingface.co/{HF_REPO_GGUF}")
print(f"  Ollama: ollama run hf.co/{HF_REPO_GGUF}")
