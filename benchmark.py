"""
ScholarQABench inference for fine-tuned Gemma-4 E4B model.

Usage:
    python benchmark.py --task scifact
    python benchmark.py --task scifact pubmedqa qasa
    python benchmark.py --task all
    python benchmark.py --task scifact --base        # also run base (non-LoRA) model
    python benchmark.py --task scifact --batch-size 4
    python benchmark.py --task scifact --limit 20   # debug: only run 20 examples

Outputs predictions to ./predictions/<task>_finetuned.jsonl (and _base.jsonl with --base).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.benchmark_config import BENCHMARK_TASKS, BenchmarkTask
from src.config import load_config
from src.model import load_model_and_tokenizer
from src.prompt_templates import build_prompt
from src.answer_parser import normalize_scifact, normalize_pubmedqa


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ScholarQABench inference")
    parser.add_argument(
        "--task", nargs="+", default=["all"],
        help="Task name(s) to run, or 'all'. Choices: " + ", ".join(BENCHMARK_TASKS),
    )
    parser.add_argument("--base", action="store_true", help="Also generate with the base model (no LoRA)")
    parser.add_argument("--batch-size", type=int, default=4, help="Generation batch size")
    parser.add_argument("--limit", type=int, default=None, help="Max examples per task (debug mode)")
    parser.add_argument("--output-dir", default="predictions", help="Directory for prediction files")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--lora-dir", default=None, help="Path to LoRA adapter dir (default: cfg.output_dir)")
    return parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(task: BenchmarkTask) -> list[dict]:
    path = task.abs_data_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Set SCHOLARQABENCH_DIR env var or clone https://github.com/AkariAsai/ScholarQABench"
        )

    if task.data_format == "json":
        with open(path) as f:
            raw = json.load(f)
        # ScholarQA-CS is a list of dicts with initial_prompt, case_id, etc.
        if isinstance(raw, list):
            return [{"input": item["initial_prompt"], "_case_id": item.get("case_id", "")} for item in raw]
        else:
            return [{"input": item["initial_prompt"], "_case_id": item.get("case_id", "")} for item in raw.values()]
    else:
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items


# ── Context extraction ────────────────────────────────────────────────────────

def _ctx_list(item: dict, task: BenchmarkTask) -> list[dict]:
    """Return the list of context dicts to include in the prediction output."""
    if task.task_type == "claim_verification" or task.task_type == "yesno_qa":
        gold_ctx = item.get("gold_ctx")
        if not gold_ctx:
            return []
        if isinstance(gold_ctx, list):
            return [{"title": c.get("title", ""), "text": c.get("text", "")} for c in gold_ctx if isinstance(c, dict)]
        return [{"title": gold_ctx.get("title", ""), "text": gold_ctx.get("text", "")}]
    elif task.task_type == "longform_qa":
        ctxs = item.get("ctxs", [])
        gold_indices = item.get("gold_ctxs", [])
        result = []
        for idx in gold_indices:
            if 0 <= idx < len(ctxs):
                c = ctxs[idx]
                result.append({"title": c.get("title", ""), "text": c.get("text", "")})
        return result
    else:
        # synthesis tasks: no retrieval
        return []


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    messages_batch = [
        [{"role": "user", "content": [{"type": "text", "text": p}]}]
        for p in prompts
    ]

    inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            use_cache=True,
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[:, input_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [s.strip() for s in decoded]


def postprocess_output(output: str, task: BenchmarkTask) -> str:
    if task.task_type == "claim_verification":
        return normalize_scifact(output)
    elif task.task_type == "yesno_qa":
        return normalize_pubmedqa(output)
    return output


def run_task(
    model,
    tokenizer,
    task: BenchmarkTask,
    items: list[dict],
    batch_size: int,
    suffix: str,
    output_dir: str,
) -> str:
    """Run inference for one task. Returns path to the prediction file."""
    out_path = os.path.join(output_dir, f"{task.name}_{suffix}.jsonl")

    predictions = []
    t0 = time.time()

    for i in tqdm(range(0, len(items), batch_size), desc=f"{task.name} ({suffix})"):
        batch = items[i : i + batch_size]
        prompts = [build_prompt(item, task) for item in batch]
        outputs = generate_batch(model, tokenizer, prompts, task.max_new_tokens)

        for item, output in zip(batch, outputs):
            output = postprocess_output(output, task)
            pred = {
                "input": item["input"],
                "question": item["input"],   # alias used by some eval scripts
                "output": output,
                "ctxs": _ctx_list(item, task),
                "answer": item.get("answer", ""),
            }
            if "_case_id" in item:
                pred["case_id"] = item["_case_id"]
            predictions.append(pred)

    elapsed = round((time.time() - t0) / 60, 1)
    print(f"  Done: {len(predictions)} predictions in {elapsed} min → {out_path}")

    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    return out_path


# ── Adapter toggle helpers ────────────────────────────────────────────────────

def _enable_adapter(model):
    """Enable LoRA adapter (handles both PEFT API variants)."""
    if hasattr(model, "enable_adapters"):
        model.enable_adapters()
    elif hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()


def _disable_adapter(model):
    """Disable LoRA adapter (handles both PEFT API variants)."""
    if hasattr(model, "disable_adapters"):
        model.disable_adapters()
    elif hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()


# ── Model loading helpers ─────────────────────────────────────────────────────

def load_finetuned_model(cfg, lora_dir: str | None):
    """Load base model + attach LoRA adapter."""
    model, tokenizer = load_model_and_tokenizer(cfg)
    adapter_path = lora_dir or cfg.output_dir
    if not os.path.exists(adapter_path):
        print(f"WARNING: LoRA adapter not found at {adapter_path}. Using base model only.")
        return model, tokenizer, False
    print(f"Loading LoRA adapter from {adapter_path}...")
    model.load_adapter(adapter_path)
    return model, tokenizer, True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config, skip_gguf=True)

    task_names = list(BENCHMARK_TASKS.keys()) if "all" in args.task else args.task
    for name in task_names:
        if name not in BENCHMARK_TASKS:
            print(f"ERROR: Unknown task '{name}'. Choices: {list(BENCHMARK_TASKS.keys())}")
            sys.exit(1)

    print(f"Tasks: {task_names}")
    print(f"Batch size: {args.batch_size}")
    if args.limit:
        print(f"Limit: {args.limit} examples per task (debug mode)")

    # Load model once, reuse across all tasks
    model, tokenizer, has_adapter = load_finetuned_model(cfg, args.lora_dir)

    for name in task_names:
        task = BENCHMARK_TASKS[name]
        print(f"\n{'='*60}")
        print(f"Task: {task.name} | Tier {task.tier} | {task.task_type}")
        print(f"Data: {task.abs_data_path}")

        items = load_data(task)
        if args.limit:
            items = items[: args.limit]
        print(f"Examples: {len(items)}")

        # Fine-tuned model predictions
        if has_adapter:
            _enable_adapter(model)
        run_task(model, tokenizer, task, items, args.batch_size, "finetuned", args.output_dir)

        # Base model predictions (optional)
        if args.base and has_adapter:
            _disable_adapter(model)
            run_task(model, tokenizer, task, items, args.batch_size, "base", args.output_dir)
            _enable_adapter(model)

    print(f"\nAll done. Predictions saved to ./{args.output_dir}/")
    print("Next step: python evaluate.py --task " + " ".join(task_names))


if __name__ == "__main__":
    main()
