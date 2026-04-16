"""
Agentic TikZ generation with iterative visual refinement.

The agent generates TikZ code, compiles it, inspects the rendered image,
and refines iteratively until the figure looks correct or max iterations
are reached.

Two backends:
  --backend transformers  (lab GPU, uses model + LoRA adapter directly)
  --backend ollama        (M1 Pro Mac, uses Ollama API)

Usage:
    # Transformers backend (on the lab GPU after training)
    python -m tikz.inference \\
        --description "A flowchart showing the training pipeline..." \\
        --backend transformers \\
        --adapter ./gemma4-4b-tikz-lora

    # Ollama backend (on M1 Pro Mac)
    python -m tikz.inference \\
        --description "A graph showing accuracy over epochs..." \\
        --backend ollama \\
        --model gemma4-4b-tikz  # your Ollama model name

    # Pipe a description from stdin
    echo "A Venn diagram of sets A and B" | python -m tikz.inference --backend ollama
"""
import argparse
import os
import sys
import shutil
import textwrap
from dataclasses import dataclass, field
from typing import Optional, List

from tikz.compile_tikz import compile_tikz, CompileResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RefinementStep:
    iteration: int
    generated_code: str
    compile_result: CompileResult
    feedback: str = ""  # What was wrong / what to fix


@dataclass
class AgentResult:
    final_code: str
    final_png: Optional[str]
    success: bool
    iterations: int
    steps: List[RefinementStep] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _initial_prompt(description: str) -> str:
    return (
        f"Generate complete, compilable TikZ/LaTeX code for the following scientific figure:\n\n"
        f"{description}"
    )


def _compile_error_prompt(description: str, code: str, error: str) -> str:
    return (
        f"The following TikZ/LaTeX code failed to compile.\n\n"
        f"Original figure description:\n{description}\n\n"
        f"Failed code:\n```latex\n{code}\n```\n\n"
        f"Compilation error:\n{error}\n\n"
        f"Fix the code so it compiles correctly. Return only the complete, fixed LaTeX document."
    )


def _visual_refinement_prompt(description: str, code: str, critique: str) -> str:
    return (
        f"The TikZ code compiled but the rendered figure has issues.\n\n"
        f"Original description:\n{description}\n\n"
        f"Current code:\n```latex\n{code}\n```\n\n"
        f"Issues observed:\n{critique}\n\n"
        f"Generate improved TikZ/LaTeX code that fixes these issues. "
        f"Return only the complete, corrected LaTeX document."
    )


def _image_inspection_prompt(description: str) -> str:
    return (
        f"This is a rendered TikZ figure. The original description was:\n\n"
        f"{description}\n\n"
        f"Carefully compare the rendered figure to the description. "
        f"List any issues: overlapping text, incorrect layout, missing elements, "
        f"wrong labels, visual clutter, or anything that doesn't match the description. "
        f"If the figure looks correct, say 'LOOKS CORRECT'. "
        f"Otherwise, describe the specific issues concisely."
    )


# ---------------------------------------------------------------------------
# Transformers backend
# ---------------------------------------------------------------------------

def _generate_transformers(model, tokenizer, prompt: str, max_new_tokens: int = 2048) -> str:
    import torch
    from unsloth import FastModel

    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    FastModel.for_inference(model)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def _inspect_image_transformers(model, tokenizer, png_path: str, prompt: str) -> str:
    """Use vision capabilities to inspect the rendered image."""
    import torch
    from PIL import Image
    from unsloth import FastModel

    image = Image.open(png_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    FastModel.for_inference(model)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _generate_ollama(model_name: str, prompt: str, max_tokens: int = 2048) -> str:
    try:
        import ollama
    except ImportError:
        raise ImportError("Install ollama: pip install ollama")

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens, "temperature": 0.1, "top_p": 0.95},
    )
    return response["message"]["content"].strip()


def _inspect_image_ollama(
    model_name: str,
    png_path: str,
    prompt: str,
    vision_model: str = "llava",
) -> str:
    """
    Use a vision-capable Ollama model to inspect the rendered image.
    Falls back to the TikZ model itself if it supports vision in Ollama.
    """
    try:
        import ollama
    except ImportError:
        raise ImportError("Install ollama: pip install ollama")

    # Try using the TikZ model with vision first; fall back to llava
    for vm in [model_name, vision_model]:
        try:
            response = ollama.chat(
                model=vm,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [png_path],
                }],
                options={"num_predict": 512, "temperature": 0.1},
            )
            return response["message"]["content"].strip()
        except Exception:
            continue

    return "Unable to inspect image: no vision-capable model available in Ollama."


# ---------------------------------------------------------------------------
# Core agentic loop
# ---------------------------------------------------------------------------

def run_agentic_loop(
    description: str,
    backend: str = "transformers",
    max_iterations: int = 3,
    compile_timeout: int = 30,
    # Transformers-specific
    model=None,
    tokenizer=None,
    # Ollama-specific
    ollama_model: str = "gemma4-4b-tikz",
    ollama_vision_model: str = "llava",
    verbose: bool = True,
) -> AgentResult:
    """
    Main agentic loop: generate → compile → inspect → refine.

    Args:
        description:          Text description of the figure to generate
        backend:              "transformers" or "ollama"
        max_iterations:       Maximum refinement iterations
        compile_timeout:      Seconds to allow pdflatex per attempt
        model/tokenizer:      Required for transformers backend
        ollama_model:         Ollama model name for TikZ generation
        ollama_vision_model:  Ollama model name for image inspection (fallback)
        verbose:              Print progress to stdout

    Returns:
        AgentResult with final code, PNG path, and iteration history
    """
    def _log(msg):
        if verbose:
            print(msg)

    def _generate(prompt, max_tokens=2048):
        if backend == "transformers":
            return _generate_transformers(model, tokenizer, prompt, max_tokens)
        else:
            return _generate_ollama(ollama_model, prompt, max_tokens)

    def _inspect_image(png_path, prompt):
        if backend == "transformers":
            return _inspect_image_transformers(model, tokenizer, png_path, prompt)
        else:
            return _inspect_image_ollama(ollama_model, png_path, prompt, ollama_vision_model)

    steps = []
    current_code = None
    current_png = None
    prompt = _initial_prompt(description)

    _log(f"\n{'='*60}")
    _log(f"TikZ Agentic Loop | backend={backend} | max_iter={max_iterations}")
    _log(f"Description: {description[:100]}...")
    _log(f"{'='*60}\n")

    for iteration in range(1, max_iterations + 1):
        _log(f"[Iter {iteration}/{max_iterations}] Generating TikZ code...")

        # Generate
        current_code = _generate(prompt)

        # Clean up: extract code block if wrapped in markdown
        if "```latex" in current_code:
            start = current_code.find("```latex") + 8
            end = current_code.find("```", start)
            current_code = current_code[start:end].strip()
        elif "```" in current_code:
            start = current_code.find("```") + 3
            end = current_code.find("```", start)
            current_code = current_code[start:end].strip()

        _log(f"  Generated {len(current_code)} chars of TikZ code")

        # Compile
        _log(f"  Compiling...")
        compile_result = compile_tikz(current_code, timeout=compile_timeout)
        step = RefinementStep(iteration=iteration, generated_code=current_code, compile_result=compile_result)

        if not compile_result.success:
            error = compile_result.error_msg or "Unknown compilation error"
            _log(f"  FAILED: {error[:100]}")
            step.feedback = f"Compilation error: {error}"
            steps.append(step)

            if iteration < max_iterations:
                prompt = _compile_error_prompt(description, current_code, error)
            continue

        _log(f"  Compiled successfully!")
        current_png = compile_result.png_path

        # Inspect the rendered image (if PNG available)
        if current_png and os.path.exists(current_png):
            _log(f"  Inspecting rendered image...")
            inspection_prompt = _image_inspection_prompt(description)
            critique = _inspect_image(current_png, inspection_prompt)
            _log(f"  Critique: {critique[:150]}")

            step.feedback = critique
            steps.append(step)

            if "LOOKS CORRECT" in critique.upper() or iteration == max_iterations:
                _log(f"\n  {'DONE' if 'LOOKS CORRECT' in critique.upper() else 'MAX ITERATIONS REACHED'}")
                break

            # Build refinement prompt for next iteration
            prompt = _visual_refinement_prompt(description, current_code, critique)
        else:
            # No PNG (pdf2image not available) — still mark success
            step.feedback = "Compiled but no PNG available for visual inspection"
            steps.append(step)
            _log(f"  No PNG available, stopping after successful compilation.")
            break

    success = compile_result.success if steps else False
    final_png = current_png if success else None

    _log(f"\n{'='*60}")
    _log(f"Final result: {'SUCCESS' if success else 'FAILED'} after {len(steps)} iterations")
    if final_png:
        _log(f"Output PNG: {final_png}")
    _log(f"{'='*60}\n")

    return AgentResult(
        final_code=current_code or "",
        final_png=final_png,
        success=success,
        iterations=len(steps),
        steps=steps,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Agentic TikZ generation with iterative refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python -m tikz.inference --description "A simple flowchart" --backend ollama
          echo "A Venn diagram" | python -m tikz.inference --backend ollama --model my-tikz-model
          python -m tikz.inference --description "..." --backend transformers --adapter ./gemma4-4b-tikz-lora
        """),
    )
    parser.add_argument("--description", "-d", type=str,
                        help="Figure description (reads from stdin if omitted)")
    parser.add_argument("--backend", choices=["transformers", "ollama"], default="ollama",
                        help="Inference backend (default: ollama)")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Max refinement iterations (default: 3)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="pdflatex timeout per attempt in seconds (default: 30)")
    parser.add_argument("--output", "-o", type=str,
                        help="Save final TikZ code to this file")

    # Transformers-specific
    parser.add_argument("--config", default="tikz_config.yaml",
                        help="TikZ config YAML (for transformers backend)")
    parser.add_argument("--adapter", type=str,
                        help="LoRA adapter path (overrides config output_dir)")

    # Ollama-specific
    parser.add_argument("--model", type=str, default="gemma4-4b-tikz",
                        help="Ollama model name (default: gemma4-4b-tikz)")
    parser.add_argument("--vision-model", type=str, default="llava",
                        help="Ollama vision model for image inspection (default: llava)")

    args = parser.parse_args()

    # Get description from args or stdin
    if args.description:
        description = args.description
    elif not sys.stdin.isatty():
        description = sys.stdin.read().strip()
    else:
        print("Error: provide --description or pipe description via stdin")
        sys.exit(1)

    model = tokenizer = None

    if args.backend == "transformers":
        from tikz.config import load_tikz_config
        from tikz.model import load_model_and_tokenizer, attach_lora

        cfg = load_tikz_config(args.config)
        adapter_path = args.adapter or cfg.output_dir

        print(f"Loading model {cfg.model_name}...")
        model, tokenizer = load_model_and_tokenizer(cfg)
        model = attach_lora(model, cfg)
        if os.path.isdir(adapter_path):
            model.load_adapter(adapter_path)
            print(f"Loaded adapter from {adapter_path}")
        else:
            print(f"Warning: adapter not found at {adapter_path}, using base model")

    result = run_agentic_loop(
        description=description,
        backend=args.backend,
        max_iterations=args.max_iterations,
        compile_timeout=args.timeout,
        model=model,
        tokenizer=tokenizer,
        ollama_model=args.model,
        ollama_vision_model=args.vision_model,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(result.final_code)
        print(f"Final TikZ code saved to {args.output}")
    else:
        print("\n--- Final TikZ Code ---")
        print(result.final_code[:1000] + ("..." if len(result.final_code) > 1000 else ""))

    if result.final_png:
        print(f"\nRendered PNG: {result.final_png}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
