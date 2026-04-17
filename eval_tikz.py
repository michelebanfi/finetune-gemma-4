#!/usr/bin/env python3
"""
TikZ model evaluation on the DaTikZ v3 test split.

Run with:
    python3 eval_tikz.py                          # full eval (542 examples)
    python3 eval_tikz.py --limit 50               # quick eval on first 50 examples
    python3 eval_tikz.py --compare                # also run base model for comparison
    python3 eval_tikz.py --output-dir ./my_results
Config:   tikz/config.yaml  |  Adapter: gemma4-4b-tikz-lora/
"""
import argparse
import os
import sys

from dotenv import load_dotenv
from common.bootstrap import install_tikz_eval_dependencies

load_dotenv()

# ── Bootstrap ────────────────────────────────────────────────────────────────
install_tikz_eval_dependencies()

# ── Imports ───────────────────────────────────────────────────────────────────
from tikz.config import load_tikz_config
from tikz.model import load_model_and_tokenizer
from tikz.evaluation import run_evaluation, run_qualitative_comparison

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None,
                    help="Limit number of test examples (default: all 542)")
parser.add_argument("--compare", action="store_true",
                    help="Also run base model (no LoRA) for qualitative comparison")
parser.add_argument("--output-dir", default="./tikz_results",
                    help="Directory to save results (default: ./tikz_results)")
parser.add_argument("--lora-dir", default=None,
                    help="Override LoRA adapter path (default: cfg.output_dir)")
parser.add_argument("--config", default="tikz/config.yaml",
                    help="Path to TikZ config YAML (default: tikz/config.yaml)")
args = parser.parse_args()

# ── Config + Model ────────────────────────────────────────────────────────────
cfg = load_tikz_config(args.config)
adapter_path = args.lora_dir or cfg.output_dir
print(f"Model:   {cfg.model_name}")
print(f"Adapter: {adapter_path}")

# Check adapter exists
if not os.path.isdir(adapter_path):
    print(f"\nERROR: LoRA adapter not found at {adapter_path}")
    print("Run train_tikz.py first to train the model.")
    sys.exit(1)

model, tokenizer = load_model_and_tokenizer(cfg)

# Load adapter (PEFT attaches + loads weights in one step)
print(f"Loading LoRA adapter from {adapter_path}...")
model.load_adapter(adapter_path)

# For qualitative comparison: keep a reference to the base model (adapter disabled)
if args.compare:
    def _enable_adapter(m):
        if hasattr(m, "enable_adapters"): m.enable_adapters()
        elif hasattr(m, "enable_adapter_layers"): m.enable_adapter_layers()

    def _disable_adapter(m):
        if hasattr(m, "disable_adapters"): m.disable_adapters()
        elif hasattr(m, "disable_adapter_layers"): m.disable_adapter_layers()

    base_model = model  # Same model; toggle adapter on/off
    _disable_adapter(base_model)

# ── Evaluation ────────────────────────────────────────────────────────────────
print(f"\nRunning evaluation on {cfg.eval_dataset} test split...")
metrics = run_evaluation(
    model=model,
    tokenizer=tokenizer,
    cfg=cfg,
    output_dir=args.output_dir,
    limit=args.limit,
)

print(f"\nCompilation success rate: {metrics['compile_success_rate']:.1f}%")
print(f"  ({metrics['n_compiled']}/{metrics['n_total']} examples compiled)")

# ── Optional qualitative comparison ──────────────────────────────────────────
if args.compare:
    print("\nRunning qualitative comparison (base vs finetuned)...")
    # base_model = adapter disabled, finetuned = adapter enabled
    _enable_adapter(model)  # make sure finetuned is active for comparison
    run_qualitative_comparison(
        base_model=base_model,        # same model object, adapter will be toggled inside
        finetuned_model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        output_dir=args.output_dir,
    )
