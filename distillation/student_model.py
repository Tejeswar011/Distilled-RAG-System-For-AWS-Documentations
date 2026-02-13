#Note: This Part is trained on "Google Colab"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load dataset
dataset = load_dataset("json", data_files="data/distillation_dataset/dataset.jsonl")

def format_example(example):
    return {
        "text": f"Question: {example['question']}\nAnswer: {example['answer']}"
    }

dataset = dataset.map(format_example)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Model (no fp16)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

# Training config (disable mixed precision)
training_args = SFTConfig(
    output_dir="models",
    per_device_train_batch_size=1,   # safer for Colab GPU
    num_train_epochs=2,
    logging_steps=10,
    save_steps=200,
    fp16=False,
    bf16=False,
    dataset_text_field="text"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args
)

trainer.train()

trainer.save_model("models")
tokenizer.save_pretrained("models")