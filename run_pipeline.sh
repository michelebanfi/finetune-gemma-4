#!/usr/bin/env bash
# Full pipeline: fine-tune → GGUF conversion → HuggingFace upload
#
# Runs training and conversion as separate Python processes to avoid the
# transformers version conflict (training needs 5.5.0, llama.cpp converter
# needs 5.3.0 for Gemma-4). convert_and_upload_gguf.py handles the
# downgrade/restore automatically.
#
# Usage:
#   ./run_pipeline.sh            # full run
#   ./run_pipeline.sh --gguf-only  # skip training, re-run conversion/upload
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

GGUF_ONLY=0
for arg in "$@"; do
    [ "$arg" = "--gguf-only" ] && GGUF_ONLY=1
done

if [ "$GGUF_ONLY" -eq 0 ]; then
    echo "================================================================"
    echo "  Step 1 of 2: Fine-tuning (GGUF export delegated to step 2)"
    echo "================================================================"
    python3 finetune_gemma4_scientific.py --skip-gguf
    echo ""
fi

echo "================================================================"
echo "  Step 2 of 2: GGUF conversion + HuggingFace upload"
echo "================================================================"
python3 convert_and_upload_gguf.py

echo ""
echo "================================================================"
echo "  Pipeline complete."
echo "================================================================"
