#!/usr/bin/env bash
# Scientific pipeline: fine-tune → merge → GGUF conversion → HuggingFace upload
#
# Usage:
#   ./run_pipeline.sh             # full run (train + merge + GGUF + upload)
#   ./run_pipeline.sh --gguf-only # skip training; re-convert + upload from existing safetensors
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

GGUF_ONLY=0
for arg in "$@"; do
    [ "$arg" = "--gguf-only" ] && GGUF_ONLY=1
done

if [ "$GGUF_ONLY" -eq 0 ]; then
    echo "================================================================"
    echo "  Fine-tuning + merge + GGUF conversion + HuggingFace upload"
    echo "================================================================"
    python3 train.py
else
    echo "================================================================"
    echo "  GGUF conversion + HuggingFace upload (skipping training)"
    echo "================================================================"
    python3 export_gguf.py
fi

echo ""
echo "================================================================"
echo "  Pipeline complete."
echo "================================================================"
