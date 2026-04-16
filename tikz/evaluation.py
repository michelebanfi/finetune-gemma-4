"""
TikZ model evaluation on the DaTikZ v3 test split (542 examples).

Metrics:
  1. Compilation success rate  — primary metric (target >70%)
  2. LPIPS / SSIM              — visual similarity for successful compilations
  3. Qualitative side-by-side  — 10 examples, base vs finetuned
"""
import os
import json
import time
import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from unsloth import FastModel

from tikz.config import TikZConfig
from tikz.compile_tikz import compile_tikz, check_dependencies

# Optional: visual metrics
try:
    import lpips
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    _HAS_VISUAL_METRICS = True
except ImportError:
    _HAS_VISUAL_METRICS = False


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_tikz(model, tokenizer, caption: str, max_new_tokens: int = 2048) -> str:
    """Generate TikZ code from a caption using the finetuned model."""
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Generate complete, compilable TikZ/LaTeX code for the following scientific figure:\n\n{caption}"}
            ],
        }
    ]
    input_text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated.strip()


# ---------------------------------------------------------------------------
# Visual similarity
# ---------------------------------------------------------------------------

def _compute_visual_similarity(generated_png: str, reference_png_or_image) -> dict:
    """Compute LPIPS and SSIM between generated and reference images."""
    if not _HAS_VISUAL_METRICS:
        return {}

    try:
        gen_img = Image.open(generated_png).convert("RGB")

        if isinstance(reference_png_or_image, str):
            ref_img = Image.open(reference_png_or_image).convert("RGB")
        else:
            # PIL Image from the dataset
            ref_img = reference_png_or_image.convert("RGB")

        # Resize to same size for comparison
        target_size = (256, 256)
        gen_img = gen_img.resize(target_size)
        ref_img = ref_img.resize(target_size)

        gen_arr = np.array(gen_img)
        ref_arr = np.array(ref_img)

        # SSIM
        ssim_score = ssim(gen_arr, ref_arr, channel_axis=2, data_range=255)

        # LPIPS (normalize to [-1, 1])
        lpips_fn = lpips.LPIPS(net="alex", verbose=False)
        gen_t = torch.tensor(gen_arr).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
        ref_t = torch.tensor(ref_arr).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
        lpips_score = float(lpips_fn(gen_t, ref_t).item())

        return {"ssim": ssim_score, "lpips": lpips_score}

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    model,
    tokenizer,
    cfg: TikZConfig,
    output_dir: str = "./tikz_results",
    limit: Optional[int] = None,
):
    """
    Evaluate the model on the DaTikZ v3 test split.

    Args:
        model:      Loaded model (finetuned)
        tokenizer:  Loaded tokenizer
        cfg:        TikZ config
        output_dir: Directory to save results
        limit:      Limit number of test examples (None = all 542)
    """
    deps = check_dependencies()
    print(f"Compilation dependencies: {deps}")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load test set
    print(f"\nLoading {cfg.eval_dataset} test split...")
    test_data = load_dataset(cfg.eval_dataset, split="test")
    if limit:
        test_data = test_data.select(range(min(limit, len(test_data))))
    print(f"Evaluating on {len(test_data)} examples")

    results = []
    compile_successes = 0
    visual_scores = []

    FastModel.for_inference(model)

    for i, row in enumerate(test_data):
        caption = row.get("caption", "")
        reference_image = row.get("image")  # PIL Image in datikz-v3
        reference_code = row.get("code", "")

        print(f"[{i+1}/{len(test_data)}] Generating...", end=" ", flush=True)
        t0 = time.time()

        generated_code = _generate_tikz(model, tokenizer, caption)
        gen_time = round(time.time() - t0, 2)

        # Compile
        compile_result = compile_tikz(generated_code, timeout=30)
        compile_successes += int(compile_result.success)

        status = "OK" if compile_result.success else "FAIL"
        print(f"{status} ({gen_time}s)")

        entry = {
            "index": i,
            "caption": caption[:200],
            "generated_code": generated_code,
            "compile_success": compile_result.success,
            "compile_error": compile_result.error_msg,
            "gen_time_s": gen_time,
        }

        # Visual similarity for successful compilations
        if compile_result.success and compile_result.png_path and reference_image:
            scores = _compute_visual_similarity(compile_result.png_path, reference_image)
            entry.update(scores)
            if "ssim" in scores:
                visual_scores.append(scores)

        results.append(entry)

    # Summary
    n = len(results)
    compile_rate = compile_successes / n * 100
    print(f"\n{'='*60}")
    print(f"Compilation success rate: {compile_successes}/{n} ({compile_rate:.1f}%)")

    if visual_scores:
        mean_ssim = sum(s["ssim"] for s in visual_scores) / len(visual_scores)
        mean_lpips = sum(s["lpips"] for s in visual_scores) / len(visual_scores)
        print(f"Visual similarity (on {len(visual_scores)} compiled examples):")
        print(f"  Mean SSIM:  {mean_ssim:.4f}")
        print(f"  Mean LPIPS: {mean_lpips:.4f}")
    print(f"{'='*60}\n")

    # Save results
    results_path = os.path.join(output_dir, f"eval_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "model": cfg.model_name,
            "n_examples": n,
            "compile_success_rate": round(compile_rate, 2),
            "mean_ssim": round(sum(s["ssim"] for s in visual_scores) / len(visual_scores), 4) if visual_scores else None,
            "mean_lpips": round(sum(s["lpips"] for s in visual_scores) / len(visual_scores), 4) if visual_scores else None,
            "examples": results,
        }, f, indent=2)
    print(f"Results saved to {results_path}")

    return {
        "compile_success_rate": compile_rate,
        "n_compiled": compile_successes,
        "n_total": n,
    }


# ---------------------------------------------------------------------------
# Qualitative side-by-side
# ---------------------------------------------------------------------------

def _toggle_adapter(model, enable: bool):
    if enable:
        if hasattr(model, "enable_adapters"): model.enable_adapters()
        elif hasattr(model, "enable_adapter_layers"): model.enable_adapter_layers()
    else:
        if hasattr(model, "disable_adapters"): model.disable_adapters()
        elif hasattr(model, "disable_adapter_layers"): model.disable_adapter_layers()


def run_qualitative_comparison(
    base_model,
    finetuned_model,
    tokenizer,
    cfg: TikZConfig,
    output_dir: str = "./tikz_results",
    n_examples: int = 10,
):
    """
    Generate TikZ from the same captions with both base and finetuned model,
    compile both, and save side-by-side results.

    base_model and finetuned_model may be the same object — adapter is toggled
    on/off between comparisons (same approach as benchmark.py).
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    test_data = load_dataset(cfg.eval_dataset, split="test")
    # Pick a diverse sample spread across the test set
    indices = list(range(0, len(test_data), max(1, len(test_data) // n_examples)))[:n_examples]
    samples = test_data.select(indices)

    output_lines = [f"TikZ Qualitative Comparison — {timestamp}\n{'='*70}\n"]

    same_model = base_model is finetuned_model

    for i, row in enumerate(samples):
        caption = row.get("caption", "")
        output_lines.append(f"\n[{i+1}/{n_examples}] Caption:\n{caption[:300]}\n")
        output_lines.append("-" * 40)

        for label, use_adapter in [("BASE", False), ("FINETUNED", True)]:
            model = finetuned_model  # always use the same model object
            if same_model:
                _toggle_adapter(model, enable=use_adapter)

            FastModel.for_inference(model)
            code = _generate_tikz(model, tokenizer, caption, max_new_tokens=1024)
            result = compile_tikz(code, timeout=20)
            status = "COMPILED" if result.success else f"FAILED: {result.error_msg}"
            output_lines.append(f"\n{label} [{status}]:\n{code[:500]}...\n")

    # Re-enable adapter at the end
    if same_model:
        _toggle_adapter(finetuned_model, enable=True)

    report = "\n".join(output_lines)
    report_path = os.path.join(output_dir, f"qualitative_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Qualitative comparison saved to {report_path}")
    return report_path
