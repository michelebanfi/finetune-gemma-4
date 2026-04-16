"""
DaTikZ v4 dataset loading and multi-task conversation formatting.

Three training tasks built from the same dataset rows:
  - caption_to_code:       caption (text) → tikz_code
  - image_to_description:  png_image → vlm_description
  - image_to_code:         png_image + caption → tikz_code
"""
import random
from typing import List, Dict, Any

from datasets import load_dataset, Dataset

from tikz.config import TikZConfig


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CAPTION_TO_CODE_PROMPT = (
    "Generate complete, compilable TikZ/LaTeX code for the following scientific figure:\n\n{caption}"
)

_IMAGE_TO_DESCRIPTION_PROMPT = (
    "Describe this scientific figure in detail. "
    "Include the type of diagram, its components, labels, and any "
    "mathematical or scientific concepts it illustrates."
)

_IMAGE_TO_CODE_PROMPT = (
    "Generate complete, compilable TikZ/LaTeX code that reproduces this scientific figure.\n\n"
    "Description: {caption}"
)


# ---------------------------------------------------------------------------
# Per-task conversation builders
# ---------------------------------------------------------------------------

def _make_caption_to_code(row: Dict[str, Any]) -> List[Dict]:
    """Text-only: caption → tikz_code."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _CAPTION_TO_CODE_PROMPT.format(caption=row["caption"])},
            ],
        },
        {
            "role": "model",
            "content": [
                {"type": "text", "text": row["tikz_code"]},
            ],
        },
    ]


def _make_image_to_description(row: Dict[str, Any]) -> List[Dict]:
    """Multimodal: image → vlm_description."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["png_image"]},
                {"type": "text", "text": _IMAGE_TO_DESCRIPTION_PROMPT},
            ],
        },
        {
            "role": "model",
            "content": [
                {"type": "text", "text": row["vlm_description"]},
            ],
        },
    ]


def _make_image_to_code(row: Dict[str, Any]) -> List[Dict]:
    """Multimodal: image + caption → tikz_code."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["png_image"]},
                {"type": "text", "text": _IMAGE_TO_CODE_PROMPT.format(caption=row["caption"])},
            ],
        },
        {
            "role": "model",
            "content": [
                {"type": "text", "text": row["tikz_code"]},
            ],
        },
    ]


_TASK_BUILDERS = {
    "caption_to_code": _make_caption_to_code,
    "image_to_description": _make_image_to_description,
    "image_to_code": _make_image_to_code,
}


# ---------------------------------------------------------------------------
# Task assignment
# ---------------------------------------------------------------------------

def _assign_tasks(n: int, ratios: Dict[str, float], seed: int) -> List[str]:
    """Return a list of n task names sampled according to ratios, shuffled."""
    rng = random.Random(seed)
    tasks = list(ratios.keys())
    weights = [ratios[t] for t in tasks]
    assigned = rng.choices(tasks, weights=weights, k=n)
    rng.shuffle(assigned)
    return assigned


# ---------------------------------------------------------------------------
# Dataset validation helpers
# ---------------------------------------------------------------------------

def _validate_row(row: Dict[str, Any], task: str) -> bool:
    """Return True if the row has all fields required by the task."""
    required = {
        "caption_to_code": ["caption", "tikz_code"],
        "image_to_description": ["png_image", "vlm_description"],
        "image_to_code": ["png_image", "caption", "tikz_code"],
    }
    for field in required[task]:
        val = row.get(field)
        if val is None:
            return False
        if isinstance(val, str) and not val.strip():
            return False
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_tikz_training_dataset(cfg: TikZConfig, tokenizer) -> Dataset:
    """
    Load DaTikZ v4, subsample, assign tasks, format conversations, apply
    the Gemma-4 chat template, and return a HuggingFace Dataset with a
    'text' column ready for SFTTrainer.

    For multimodal tasks the PIL images are embedded in the conversation
    before the chat template is applied, so Unsloth's processor can handle
    them. Text-only examples follow the same 'text' column format as the
    sci task.
    """
    print(f"\nLoading {cfg.dataset_name}...")
    dataset = load_dataset(cfg.dataset_name, split="train")
    print(f"Total rows: {len(dataset):,} | columns: {dataset.column_names}")

    # Subsample
    n = min(cfg.subset_size, len(dataset))
    dataset = dataset.shuffle(seed=cfg.seed).select(range(n))
    print(f"Subsampled to {len(dataset):,} rows")

    # Assign one task per row
    task_assignments = _assign_tasks(len(dataset), cfg.task_ratios, cfg.seed)

    # Count per task
    counts = {t: task_assignments.count(t) for t in cfg.task_ratios}
    print("\nTask distribution:")
    for task, count in counts.items():
        print(f"  {task}: {count:,} ({count / len(dataset) * 100:.1f}%)")

    # Build conversations
    print("\nBuilding conversations...")
    conversations = []
    skipped = 0
    for i, row in enumerate(dataset):
        task = task_assignments[i]
        if not _validate_row(row, task):
            skipped += 1
            # Fall back to caption_to_code which has the widest data coverage
            task = "caption_to_code"
            if not _validate_row(row, task):
                skipped += 1
                continue
        conversations.append(_TASK_BUILDERS[task](row))

    if skipped:
        print(f"Skipped {skipped} rows with missing/empty fields")
    print(f"Built {len(conversations):,} conversations")

    # Apply Gemma-4 chat template
    print("Applying Gemma-4 chat template...")
    texts = []
    length_filtered = 0
    for convo in conversations:
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix("<bos>")

        # Quick character-based length pre-filter (roughly 4 chars per token)
        if len(text) > cfg.max_seq_length * 4:
            length_filtered += 1
            continue
        texts.append(text)

    if length_filtered:
        print(f"Filtered {length_filtered} conversations exceeding ~{cfg.max_seq_length} tokens")
    print(f"Final training examples: {len(texts):,}")

    result = Dataset.from_dict({"text": texts})

    print("\n--- Sample (first 500 chars) ---")
    print(result[0]["text"][:500])
    print("---\n")

    return result
