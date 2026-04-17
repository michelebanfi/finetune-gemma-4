"""
Dataset loading, normalization, and formatting for Gemma-4 chat template.
"""
from datasets import load_dataset, concatenate_datasets
from unsloth.chat_templates import standardize_data_formats

from sci.config import Config


def _get_conv_col(dataset) -> str:
    if "conversations" in dataset.column_names:
        return "conversations"
    if "messages" in dataset.column_names:
        return "messages"
    raise KeyError(f"No conversation column found. Columns: {dataset.column_names}")


def _remap_roles(example):
    """Remap 'assistant' -> 'model' for Gemma-4's chat template."""
    for msg in example["conversations"]:
        if msg["role"] == "assistant":
            msg["role"] = "model"
    return {"conversations": example["conversations"]}


def build_training_dataset(cfg: Config, tokenizer):
    print("\nLoading datasets...")
    os_data = load_dataset("OpenSciLM/OS_Train_Data", split="train")
    sciriff_data = load_dataset("allenai/SciRIFF-train-mix", split="train")

    print(f"OS_Train_Data:     {len(os_data):,} rows | columns: {os_data.column_names}")
    print(f"SciRIFF-train-mix: {len(sciriff_data):,} rows | columns: {sciriff_data.column_names}")

    # Subsample
    os_data = os_data.shuffle(seed=cfg.seed).select(range(min(cfg.os_data_subset, len(os_data))))
    sciriff_data = sciriff_data.shuffle(seed=cfg.seed).select(range(min(cfg.sciriff_data_subset, len(sciriff_data))))

    print(f"\nAfter subsampling:")
    print(f"  OS_Train_Data:     {len(os_data):,} rows")
    print(f"  SciRIFF-train-mix: {len(sciriff_data):,} rows")

    # Standardize format
    os_data = standardize_data_formats(os_data)
    sciriff_data = standardize_data_formats(sciriff_data)

    # Normalize conversation column name
    os_col = _get_conv_col(os_data)
    sr_col = _get_conv_col(sciriff_data)
    print(f"Conversation column — OS: '{os_col}', SciRIFF: '{sr_col}'")

    if os_col != "conversations":
        os_data = os_data.rename_column(os_col, "conversations")
    if sr_col != "conversations":
        sciriff_data = sciriff_data.rename_column(sr_col, "conversations")

    # Remap roles
    if any(m["role"] == "assistant" for m in os_data[0]["conversations"]):
        print("Remapping 'assistant' -> 'model' roles...")
        os_data = os_data.map(_remap_roles, desc="Remapping OS roles")
        sciriff_data = sciriff_data.map(_remap_roles, desc="Remapping SciRIFF roles")
    else:
        print("Roles already use 'model'.")

    # Keep only the conversations column and merge
    os_data = os_data.remove_columns([c for c in os_data.column_names if c != "conversations"])
    sciriff_data = sciriff_data.remove_columns([c for c in sciriff_data.column_names if c != "conversations"])

    combined = concatenate_datasets([os_data, sciriff_data]).shuffle(seed=cfg.seed)
    print(f"\nTotal training examples: {len(combined):,}")

    # Apply Gemma-4 chat template
    def _format(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False,
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    print("Applying Gemma-4 chat template...")
    combined = combined.map(_format, batched=True, desc="Formatting")

    print("\n--- Sample (first 500 chars) ---")
    print(combined[0]["text"][:500])
    print("---\n")

    return combined
