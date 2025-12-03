import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen1.5-0.5B-Chat"
LORA_DIR = "qwen-uz-constitution-lora"
MAX_NEW_TOKENS = 128

device = "cpu"
print(f"Using device: {device}")

print(f"Loading base model & tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
)
base_model.to(device)

print(f"Loading LoRA adapter from {LORA_DIR} ...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model + LoRA loaded successfully ✅")


def build_prompt(user_prompt: str):
    system_prompt = "You are an assistant that answers questions about the Constitution of the Republic of Uzbekistan."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = (
            f"{system_prompt}\n\n"
            f"User: {user_prompt}\n"
            f"Assistant:"
        )
    return prompt_text


def generate_response(user_prompt: str) -> str:
    print("Generating the answer...")
    prompt_text = build_prompt(user_prompt)

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,     # deterministic for evaluation
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return decoded.strip()


if __name__ == "__main__":
    print("Ready for Constitution Q&A. Type your question:")
    question = input("You: ")
    answer = generate_response(question)
    print("Assistant:", answer)
