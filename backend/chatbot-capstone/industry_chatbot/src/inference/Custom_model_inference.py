from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./fine_tuned_model"  # Path to your fine-tuned model

# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

# Ensure the tokenizer has a padding token
tokenizer.pad_token = tokenizer.eos_token

# Generate response from a test question
input_text = "Question: Explain server monitoring policy according to the IT policy?"
inputs = tokenizer(
    input_text, return_tensors="pt", padding=True, truncation=True
).to("cuda")  # Move inputs to GPU

# Generate output with controlled randomness
output = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],  # ✅ Explicitly passing attention mask
    max_length=100,
    do_sample=True,  # Enable sampling instead of greedy search
    top_k=50,  # Keep top 50 most likely words
    top_p=0.9,  # Use nucleus sampling
    temperature=0.7,  # Lower values = more deterministic responses
    repetition_penalty=1.2,  # ✅ Reduces repetitive text
    pad_token_id=tokenizer.eos_token_id,  # ✅ Explicitly set padding token
)

# Decode and print the response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
