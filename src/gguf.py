"""
Merge LoRA adapter into base weights, convert to GGUF, and upload to HuggingFace.
"""
import os
import sys
import subprocess

from src.config import Config


def merge_and_save(model, tokenizer, cfg: Config):
    """Merge LoRA into base weights and save as fp16 safetensors."""
    print(f"\n[GGUF 1/3] Merging LoRA into base weights at {cfg.gguf_dir} ...")
    model.save_pretrained_merged(cfg.gguf_dir, tokenizer, save_method="merged_16bit")
    merged_size = sum(
        os.path.getsize(os.path.join(cfg.gguf_dir, f))
        for f in os.listdir(cfg.gguf_dir)
    )
    print(f"  Merged model saved ({round(merged_size / 1e9, 2)} GB)")


def convert_to_gguf(cfg: Config, script_dir: str) -> str:
    """Convert merged safetensors to GGUF. Returns path to the .gguf file."""
    llama_cpp_dir = os.path.join(script_dir, "llama.cpp")
    converter = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    gguf_file = os.path.join(script_dir, "model.gguf")

    if not os.path.exists(converter):
        print("\n[GGUF 2/3] Cloning llama.cpp ...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir],
            check=True,
        )
        req_file = os.path.join(llama_cpp_dir, "requirements", "requirements-convert_hf_to_gguf.txt")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], check=True)

    # llama.cpp's Gemma-4 converter requires transformers<=5.3.0;
    # downgrade for the subprocess then restore so the main process is unaffected.
    print(f"\n[GGUF 2/3] Converting to GGUF ({cfg.gguf_quantization}) → {gguf_file} ...")
    print("  Pinning transformers==5.3.0 for Gemma-4 GGUF compat ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "transformers==5.3.0"],
        check=True,
    )

    try:
        subprocess.run(
            [
                sys.executable, converter,
                cfg.gguf_dir,
                "--outfile", gguf_file,
                "--outtype", cfg.gguf_quantization.lower(),
            ],
            check=True,
        )
    finally:
        print("  Restoring transformers==5.5.0 ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "transformers==5.5.0"],
            check=True,
        )

    size_gb = round(os.path.getsize(gguf_file) / 1e9, 2)
    print(f"  GGUF saved: {gguf_file} ({size_gb} GB)")
    return gguf_file


def upload_to_hub(gguf_file: str, cfg: Config):
    """Upload GGUF file to HuggingFace Hub."""
    import shutil
    import tempfile

    if not cfg.hf_token or not cfg.hf_repo_gguf:
        print(f"\n[GGUF 3/3] Skipping upload (HF_TOKEN or hf_repo_gguf not set).")
        print(f"  To run with Ollama locally:")
        print(f"    ollama create gemma4-sci --file {gguf_file}")
        print(f"    ollama run gemma4-sci")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=cfg.hf_token)
    api.create_repo(cfg.hf_repo_gguf, repo_type="model", exist_ok=True)

    size_gb = round(os.path.getsize(gguf_file) / 1e9, 2)
    print(f"\n[GGUF 3/3] Uploading model.gguf ({size_gb} GB) to https://huggingface.co/{cfg.hf_repo_gguf} ...")
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy2(gguf_file, os.path.join(tmp, "model.gguf"))
        api.upload_large_folder(
            folder_path=tmp,
            repo_id=cfg.hf_repo_gguf,
            repo_type="model",
            allow_patterns=["model.gguf"],
        )
    print(f"  Upload complete: https://huggingface.co/{cfg.hf_repo_gguf}")
    print(f"  Run with Ollama: ollama run hf.co/{cfg.hf_repo_gguf}")
