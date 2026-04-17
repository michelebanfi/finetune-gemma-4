"""
Qualitative evaluation: base model vs fine-tuned model side-by-side comparison.
Results are streamed to stdout and saved to a timestamped file.
"""
import os
import sys
from datetime import datetime

from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template

from sci.config import Config


_QUESTIONS = [
    "What are the main mechanisms by which CRISPR-Cas9 achieves gene editing, "
    "and what are the current limitations for therapeutic applications in humans?",

    "Explain the role of transformer architectures in recent advances in protein "
    "structure prediction. How does AlphaFold2 differ from earlier approaches?",

    "A clinical trial reports a p-value of 0.03 with a sample size of 15 participants. "
    "What concerns should a reviewer raise about the statistical validity of this result?",

    "Summarize the current understanding of dark matter candidates in particle physics. "
    "What experimental approaches are being used to detect them?",

    "How do feedback loops in the climate system, such as ice-albedo feedback and water "
    "vapor feedback, amplify or dampen the effects of increased CO2 concentrations?",
]


class _Tee:
    """Duplicates stdout writes to a file."""
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


def _generate(model, tokenizer, question: str, max_new_tokens: int = 512):
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )


def run_comparison(model, tokenizer, cfg: Config):
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    results_path = os.path.join(
        cfg.output_dir,
        f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    results_file = open(results_path, "w", encoding="utf-8")
    sys.stdout = _Tee(results_file)

    print(f"Model: {cfg.model_name}  |  Adapter: {cfg.output_dir}  |  Date: {datetime.now().isoformat()}")

    for question in _QUESTIONS:
        print(f"\n{'#' * 70}")
        print(f"QUESTION: {question}")
        print(f"{'#' * 70}")

        print(f"\n--- BASE MODEL {'-' * 55}")
        model.disable_adapter_layers()
        _generate(model, tokenizer, question)

        print(f"\n--- FINE-TUNED MODEL {'-' * 49}")
        model.enable_adapter_layers()
        _generate(model, tokenizer, question)

    sys.stdout = sys.stdout._stdout
    results_file.close()
    print(f"\nTest results saved to {results_path}")
