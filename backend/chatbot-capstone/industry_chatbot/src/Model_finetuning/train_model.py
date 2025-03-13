import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to your QA dataset JSON file
data_file = r"C:\Users\moksh\classroom\chatbot_deepseek\industry_chatbot\data\knowledge_base\cleaned_augmented_qa_pairs.json"

# Load the dataset
dataset = load_dataset("json", data_files={"train": data_file})

# Load the DialoGPT model and tokenizer
model_checkpoint = "microsoft/DialoGPT-medium"  # You can use 'DialoGPT-small' or 'DialoGPT-large'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

# Move model to GPU
model.to(device)

# Ensure the tokenizer has a padding token
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

# Define preprocessing function
def preprocess_function(examples):
    inputs = [f"Question: {q}\nAnswer: {a}\n" for q, a in zip(examples["question"], examples["answer"])]
    tokenized_inputs = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    return tokenized_inputs

# Apply preprocessing
tokenized_dataset = dataset["train"].map(
    preprocess_function, batched=True, remove_columns=["question", "answer", "metadata"]
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Adjust for your GPU memory
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,  # Enable mixed precision for faster training
    report_to="none",  # Disable logging to external platforms
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
