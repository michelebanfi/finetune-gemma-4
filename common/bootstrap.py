"""Dependency bootstrap helpers shared by training/eval entrypoints."""
import subprocess
import sys


def install_train_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch>=2.8.0",
            "triton>=3.4.0",
            "torchvision",
            "bitsandbytes",
            "unsloth",
            "unsloth_zoo>=2026.4.6",
            "transformers==5.5.0",
            "torchcodec",
            "timm",
        ],
        check=True,
    )


def install_tikz_eval_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch>=2.8.0",
            "unsloth",
            "unsloth_zoo>=2026.4.6",
            "transformers==5.5.0",
            "datasets",
            "pdf2image",
            "pillow",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "lpips", "scikit-image"],
        check=False,
    )
