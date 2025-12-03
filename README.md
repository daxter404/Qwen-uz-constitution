#  Qwen Constitution Chatbot

This project is an AI chatbot fine-tuned to answer questions about the **Constitution of the Republic of Uzbekistan**.  
It uses:

- **Qwen/Qwen1.5-0.5B-Chat** (open-source model)
- **LoRA fine-tuning**
- **CPU-only** environment (no GPU required)

This README explains how to run the project **from zero**, including installing Python, setting up the environment, fine-tuning Qwen, and running the chatbot.

---

##   Requirements

###  OS
- Windows 10 / 11  
- (Works on Linux/Mac too)

###  Python
You need **Python 3.10 – 3.12**.

**Python 3.14 is too new** and some libraries may fail.

###  Download Python
https://www.python.org/downloads/


 **“Add Python to PATH”**

---

## Verify Python Installation


python --version
pip --version


If `pip` is missing:


python -m ensurepip --default-pip
python -m pip install --upgrade pip


---

##  Install Dependencies

Inside your project folder:

python -m pip install torch transformers datasets peft accelerate safetensors


This installs:

- PyTorch  
- Transformers  
- Qwen support  
- LoRA framework  
- Dataset tools  

---



## Fine-Tune Qwen with LoRA

Run the training script:


python finetune_qwen_lora.py


What this script does:

- Loads Qwen 0.5B
- Builds training prompts in chat format
- Fine-tunes LoRA on CPU
- Saves adapter to:


qwen-uz-constitution-lora/


Training takes a few minutes.

You should see:

```
Model saved to qwen-uz-constitution-lora
```

---fbf

 7. Run the Chatbot

After training:


python generate_answer.py


You’ll see:


Ready for Constitution Q&A. Type your question:
You:
```

Try:

```
What is the supreme law of Uzbekistan?
```

The response will come from your **fine-tuned** LoRA model.

