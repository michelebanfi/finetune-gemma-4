# finetune-gemma-4

Repository for multiple Gemma 4 finetuning sub-projects.

## Sub-projects

- `sci/` — Scientific literature finetuning (OpenSciLM + SciRIFF)
- `tikz/` — TikZ/LaTeX finetuning (DaTikZ)
- `common/` — Shared utilities (Unsloth model loading/LoRA helpers, trainer helpers, GGUF conversion/upload helpers, bootstrap install helpers)

## Entry points

### Scientific finetuning

- Train: `python3 train.py --config sci/config.yaml`
- Benchmark: `python3 benchmark.py --config sci/config.yaml`
- Evaluate: `python3 evaluate.py --task all --compare`
- Export GGUF: `python3 export_gguf.py --config sci/config.yaml`

### TikZ finetuning

- Train: `python3 train_tikz.py --config tikz/config.yaml`
- Evaluate: `python3 eval_tikz.py --config tikz/config.yaml`
- Agentic inference: `python3 -m tikz.inference --config tikz/config.yaml`

## Model cards

- Scientific model card: `sci/README.md`
- TikZ model card: `tikz/README.md`
