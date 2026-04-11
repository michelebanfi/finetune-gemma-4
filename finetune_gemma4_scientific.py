# ==============================================================================
# Gemma-4 31B Scientific Fine-tuning (Unsloth + QLoRA)
# Replicates the OpenSciLM approach with Gemma-4 31B on a local GPU (48+ GB VRAM).
#
# Run with: python finetune_gemma4_scientific.py
# ==============================================================================
import os

# Force console-friendly tqdm (avoids invisible HTML progress bars in some envs)
os.environ["TQDM_DISABLE"] = "0"
from tqdm.auto import tqdm

# ==============================================================================
# SECTION 1: INSTALLATION
# Install Unsloth + dependencies. Comment out if already installed.
# ==============================================================================
import subprocess, sys

def install():
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "torch>=2.8.0", "triton>=3.4.0",
        "torchvision", "bitsandbytes",
        "unsloth", "unsloth_zoo>=2026.4.6",
        "transformers==5.5.0", "torchcodec", "timm",
    ], check=True)

# Comment out the next line if packages are already installed
install()

# ==============================================================================
# SECTION 2: CONFIGURATION
# Edit config.yaml to switch models or tune hyperparameters.
# ==============================================================================
import yaml

with open("config.yaml") as _f:
    _cfg = yaml.safe_load(_f)

_preset = _cfg["presets"][_cfg["model_preset"]]
_train  = _cfg["training"]
_ds     = _cfg["dataset"]
_export = _cfg["export"]
_hub    = _cfg["hub"]

# --- Model ---
MODEL_NAME            = _preset["model_name"]
MAX_SEQ_LENGTH        = _preset["max_seq_length"]

# --- LoRA ---
LORA_R                = _preset["lora_r"]
LORA_ALPHA            = _preset["lora_alpha"]
LORA_DROPOUT          = _train["lora_dropout"]

# --- Training ---
PER_DEVICE_BATCH_SIZE = _preset["per_device_batch_size"]
GRADIENT_ACCUMULATION = _preset["gradient_accumulation"]
LEARNING_RATE         = _train["learning_rate"]
WARMUP_STEPS          = _train["warmup_steps"]
LR_SCHEDULER          = _train["lr_scheduler"]
NUM_TRAIN_EPOCHS      = _train["num_train_epochs"]
MAX_STEPS             = _train["max_steps"]
WEIGHT_DECAY          = _train["weight_decay"]
SEED                  = _train["seed"]

# --- Dataset ---
OS_DATA_SUBSET        = _ds["os_data_subset"]
SCIRIFF_DATA_SUBSET   = _ds["sciriff_data_subset"]

# --- Output ---
OUTPUT_DIR            = _preset["output_dir"]
GGUF_DIR              = _preset["gguf_dir"]

# --- Export ---
EXPORT_GGUF           = _export["gguf"]
GGUF_QUANTIZATION     = _export["gguf_quantization"]

# --- Optional: Push to Hugging Face Hub ---
HF_TOKEN              = _hub["hf_token"]
HF_REPO               = _hub["hf_repo"]
HF_REPO_GGUF          = _hub["hf_repo_gguf"]

print(f"Active preset: {_cfg['model_preset']} ({MODEL_NAME})")

# ==============================================================================
# SECTION 3: MODEL LOADING (4-bit QLoRA via Unsloth)
# ==============================================================================
import torch
from unsloth import FastModel

print(f"Loading {MODEL_NAME} in 4-bit precision...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}, {round(props.total_memory / 1e9, 1)} GB")

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    device_map="auto",   # Single GPU: puts everything on cuda:0
)

# ==============================================================================
# SECTION 4: LORA ADAPTER SETUP
# ==============================================================================
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,      # Text-only scientific QA
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    random_state=SEED,
)
model.print_trainable_parameters()

# ==============================================================================
# SECTION 5: CHAT TEMPLATE
# ==============================================================================
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-4",
)

# ==============================================================================
# SECTION 6: DATASET LOADING & PREPARATION
# ==============================================================================
from datasets import load_dataset, concatenate_datasets
from unsloth.chat_templates import standardize_data_formats

print("\nLoading datasets...")

os_data = load_dataset("OpenSciLM/OS_Train_Data", split="train")
sciriff_data = load_dataset("allenai/SciRIFF-train-mix", split="train")

print(f"OS_Train_Data: {len(os_data):,} rows | columns: {os_data.column_names}")
print(f"SciRIFF-train-mix: {len(sciriff_data):,} rows | columns: {sciriff_data.column_names}")

# Subsample
os_data = os_data.shuffle(seed=SEED).select(range(min(OS_DATA_SUBSET, len(os_data))))
sciriff_data = sciriff_data.shuffle(seed=SEED).select(range(min(SCIRIFF_DATA_SUBSET, len(sciriff_data))))

print(f"\nAfter subsampling:")
print(f"  OS_Train_Data: {len(os_data):,} rows")
print(f"  SciRIFF-train-mix: {len(sciriff_data):,} rows")

# Standardize format
os_data = standardize_data_formats(os_data)
sciriff_data = standardize_data_formats(sciriff_data)

# Detect conversation column name ('conversations' or 'messages')
def get_conv_col(dataset):
    if "conversations" in dataset.column_names:
        return "conversations"
    if "messages" in dataset.column_names:
        return "messages"
    raise KeyError(f"No conversation column found. Columns: {dataset.column_names}")

os_col = get_conv_col(os_data)
sr_col = get_conv_col(sciriff_data)
print(f"Conversation column — OS: '{os_col}', SciRIFF: '{sr_col}'")

if os_col != "conversations":
    os_data = os_data.rename_column(os_col, "conversations")
if sr_col != "conversations":
    sciriff_data = sciriff_data.rename_column(sr_col, "conversations")

# Remap "assistant" -> "model" for Gemma-4's chat template
def remap_roles(example):
    for msg in example["conversations"]:
        if msg["role"] == "assistant":
            msg["role"] = "model"
    return {"conversations": example["conversations"]}

if any(m["role"] == "assistant" for m in os_data[0]["conversations"]):
    print("Remapping 'assistant' -> 'model' roles...")
    os_data = os_data.map(remap_roles, desc="Remapping OS roles")
    sciriff_data = sciriff_data.map(remap_roles, desc="Remapping SciRIFF roles")
else:
    print("Roles already use 'model'.")

# Strip extra columns and merge
os_data = os_data.remove_columns([c for c in os_data.column_names if c != "conversations"])
sciriff_data = sciriff_data.remove_columns([c for c in sciriff_data.column_names if c != "conversations"])

combined = concatenate_datasets([os_data, sciriff_data])
combined = combined.shuffle(seed=SEED)
print(f"\nTotal training examples: {len(combined):,}")

# Apply Gemma-4 chat template
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False,
        ).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}

print("Applying Gemma-4 chat template...")
combined = combined.map(formatting_prompts_func, batched=True, desc="Formatting")

print("\n--- Sample (first 500 chars) ---")
print(combined[0]["text"][:500])
print("---\n")

# ==============================================================================
# SECTION 7: TRAINING
# ==============================================================================
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

use_bf16 = torch.cuda.is_bf16_supported()
use_fp16 = not use_bf16
print(f"Mixed precision: {'bf16' if use_bf16 else 'fp16'}")

gpu_stats = torch.cuda.get_device_properties(0)
start_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
max_memory = round(gpu_stats.total_memory / 1024**3, 2)
print(f"GPU 0: {gpu_stats.name} | VRAM: {max_memory} GB total, {start_memory} GB reserved")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=combined,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=1,
        save_strategy="steps",
        save_steps=500,
        output_dir=OUTPUT_DIR,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        seed=SEED,
        report_to="none",
        dataset_num_proc=4,
        packing=False,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)

# Verify masking
print("Verifying response masking (should see only model turns):")
decoded_labels = tokenizer.decode([
    tokenizer.pad_token_id if x == -100 else x
    for x in trainer.train_dataset[0]["labels"]
]).replace(tokenizer.pad_token, " ")
print(decoded_labels[:300])
print()

print("=" * 60)
print("Starting fine-tuning...")
print("=" * 60)
trainer_stats = trainer.train()

# Post-training stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
used_for_lora = round(used_memory - start_memory, 2)
print(f"\nTraining complete in {round(trainer_stats.metrics['train_runtime'] / 60, 1)} minutes")
print(f"Peak VRAM: {used_memory} GB / {max_memory} GB ({round(used_memory/max_memory*100, 1)}%)")
print(f"VRAM used for training: {used_for_lora} GB")

# ==============================================================================
# SECTION 8: SAVE LORA ADAPTER
# ==============================================================================
print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Adapter saved.")

if HF_TOKEN and HF_REPO:
    print(f"Pushing adapter to https://huggingface.co/{HF_REPO} ...")
    model.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    print("Push complete.")

# ==============================================================================
# SECTION 8b: EXPORT FOR OLLAMA (GGUF)
# Merges LoRA into base weights and converts to GGUF format.
# Requires ~30GB+ disk space for the merged fp16 model + GGUF output.
# ==============================================================================
if EXPORT_GGUF:
    print(f"\nExporting to GGUF at {GGUF_DIR} ({GGUF_QUANTIZATION})...")
    model.save_pretrained_gguf(
        GGUF_DIR,
        tokenizer,
        quantization_method=GGUF_QUANTIZATION,
    )
    print(f"GGUF saved to {GGUF_DIR}")

    if HF_TOKEN and HF_REPO_GGUF:
        model.push_to_hub_gguf(
            HF_REPO_GGUF,
            tokenizer,
            quantization_method=GGUF_QUANTIZATION,
            token=HF_TOKEN,
        )
        print(f"Uploaded to https://huggingface.co/{HF_REPO_GGUF}")
        print(f"Run with Ollama: ollama run hf.co/{HF_REPO_GGUF}")
    else:
        print("To run with Ollama locally:")
        print(f"  ollama create gemma4-sci -f {GGUF_DIR}/Modelfile")
        print(f"  ollama run gemma4-sci")

# ==============================================================================
# SECTION 9: TESTING — BASE vs FINE-TUNED COMPARISON
# Toggles the LoRA adapter on/off on the same model to compare responses.
# ==============================================================================
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

def _generate(question, max_new_tokens=512):
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    _ = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )

def compare(question, max_new_tokens=512):
    """Run the same question through base model and fine-tuned model back-to-back."""
    print(f"\n{'#'*70}")
    print(f"QUESTION: {question}")
    print(f"{'#'*70}")

    print(f"\n--- BASE MODEL {'-'*55}")
    model.disable_adapter_layers()
    _generate(question, max_new_tokens)

    print(f"\n--- FINE-TUNED MODEL {'-'*49}")
    model.enable_adapter_layers()
    _generate(question, max_new_tokens)

# Scientific comparison tests
compare("What are the main mechanisms by which CRISPR-Cas9 achieves gene editing, "
        "and what are the current limitations for therapeutic applications in humans?")

compare("Explain the role of transformer architectures in recent advances in protein "
        "structure prediction. How does AlphaFold2 differ from earlier approaches?")

compare("A clinical trial reports a p-value of 0.03 with a sample size of 15 participants. "
        "What concerns should a reviewer raise about the statistical validity of this result?")

compare("Summarize the current understanding of dark matter candidates in particle physics. "
        "What experimental approaches are being used to detect them?")

compare("How do feedback loops in the climate system, such as ice-albedo feedback and water "
        "vapor feedback, amplify or dampen the effects of increased CO2 concentrations?")
