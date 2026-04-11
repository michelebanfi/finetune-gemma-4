import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Start with the 31B model (QLoRA makes it fit in 96GB VRAM) or E4B for testing
MODEL_ID = "google/gemma-4-31b" 
OUTPUT_DIR = "./gemma-4-openscholar-checkpoints"

# QLoRA Parameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training Parameters optimized for 96GB VRAM
BATCH_SIZE = 4           # Adjust based on sequence length; 4-8 is safe for 96GB
GRAD_ACCUMULATION = 4    # Effective batch size = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 2048    # Scientific texts are long, ensure you have enough context

# ==============================================================================
# 2. DATASET PREPARATION
# ==============================================================================
def load_and_prep_data():
    print("Loading OpenSciLM datasets...")
    
    # 1. The Synthetic OpenScholar Data (~130k rows)
    os_data = load_dataset("OpenSciLM/OS_Train_Data", split="train")
    
    # 2. The SciRIFF + General Tulu Mix (Scientific instruction + general domain)
    # This matches the paper's 50/50 domain distribution strategy
    sciriff_mix = load_dataset("allenai/SciRIFF-train-mix", split="train")
    
    # We need to map both datasets into a standard conversational format for Gemma.
    # Gemma's template expects a list of dictionaries with 'role' and 'content'.
    
    def format_os_data(example):
        # OpenSciLM data usually has 'query', 'documents', and 'response'/'feedback'
        # Adjust these keys if OS_Train_Data column names differ slightly.
        prompt = f"Answer the query based on the provided documents.\n\nQuery: {example.get('query', '')}\n\nDocuments:\n{example.get('documents', '')}"
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "model", "content": example.get('response', '')}
            ]
        }
        
    def format_sciriff_data(example):
        # SciRIFF mix typically follows user/assistant formats natively
        # Fallback to standard prompt/completion mappings if structured differently
        return {
            "messages": [
                {"role": "user", "content": example.get('prompt', '') or example.get('messages', [{'content': ''}])[0]['content']},
                {"role": "model", "content": example.get('completion', '') or example.get('messages', [{'content': ''}])[1]['content']}
            ]
        }

    print("Formatting datasets to Gemma 4 chat template...")
    os_data = os_data.map(format_os_data, remove_columns=os_data.column_names)
    sciriff_mix = sciriff_mix.map(format_sciriff_data, remove_columns=sciriff_mix.column_names)
    
    # Combine and shuffle
    combined_dataset = concatenate_datasets([os_data, sciriff_mix])
    combined_dataset = combined_dataset.shuffle(seed=42)
    print(f"Total training examples: {len(combined_dataset)}")
    
    return combined_dataset

# ==============================================================================
# 3. MODEL & TOKENIZER INITIALIZATION (QLoRA)
# ==============================================================================
def initialize_model_and_tokenizer():
    print(f"Initializing {MODEL_ID} in 4-bit precision...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Better for causal LM training

    # Configure 4-bit quantization (QLoRA) to drastically reduce VRAM usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, # Saves additional memory
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", # Automatically distributes across your 96GB GPU
        torch_dtype=torch.bfloat16,
    )

    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable() # CRITICAL for saving memory with 31B models

    # Gemma architecture target modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================
def main():
    # 1. Load Data
    dataset = load_and_prep_data()
    
    # 2. Load Model
    model, tokenizer = initialize_model_and_tokenizer()
    
    # 3. Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        num_train_epochs=2, # OpenSciLM used 2 epochs
        optim="paged_adamw_8bit", # Specific optimizer for QLoRA memory savings
        bf16=True, # Modern GPUs (like yours) should use bf16
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none" # Set to "wandb" if you use Weights & Biases
    )

    # 4. Initialize SFTTrainer
    # The SFTTrainer will automatically apply Gemma's chat template to the "messages" column
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=get_peft_model(model, LoraConfig(task_type="CAUSAL_LM")).peft_config, # Passing the config extract
        dataset_text_field="messages", # TRL handles chat templates automatically now!
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        packing=False, # Set to True to speed up training, but can complicate retrieval context
    )

    # 5. Start Training
    print("Starting fine-tuning...")
    trainer.train()
    
    # 6. Save the final adapter weights
    print(f"Training complete. Saving adapter to {OUTPUT_DIR}/final_adapter")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")

if __name__ == "__main__":
    main()
