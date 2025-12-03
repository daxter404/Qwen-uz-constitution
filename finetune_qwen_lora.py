import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# Base chat model
BASE_MODEL = "Qwen/Qwen1.5-0.5B-Chat"
DATA_FILE = "data/uz_constitution.jsonl"
OUTPUT_DIR = "qwen-uz-constitution-lora"   # where LoRA adapter will be saved

MAX_SEQ_LEN = 512

device = "cpu"  # your PyTorch is CPU-only
print(f"Using device: {device}")

# 1. Load tokenizer & base model
print(f"Loading tokenizer for {BASE_MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print(f"Loading base model {BASE_MODEL} ...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
)
model.to(device)

# 2. Wrap with LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Load dataset
print(f"Loading dataset from {DATA_FILE} ...")
raw_datasets = load_dataset("json", data_files=DATA_FILE)

# 4. Build supervised text with chat template
def build_text(example):
    # System prompt to specialize the model
    system_prompt = "You are an assistant that answers questions about the Constitution of the Republic of Uzbekistan."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # we include both Q & A in training
        )
    else:
        text = (
            f"{system_prompt}\n\n"
            f"User: {example['instruction']}\n"
            f"Assistant: {example['output']}"
        )

    return {"text": text}

processed_datasets = raw_datasets.map(build_text)

# 5. Tokenize and create labels
def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_datasets = processed_datasets.map(
    tokenize_function,
    remove_columns=processed_datasets["train"].column_names,
)

train_dataset = tokenized_datasets["train"]

# 6. Training setup (small dataset -> many epochs, tiny batch)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=50,            # dataset is tiny; we can over-train a bit
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("Starting training ...")
trainer.train()

print("Saving LoRA adapter ...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done. LoRA saved to", OUTPUT_DIR)
