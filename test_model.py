# test_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Path to your saved fine-tuned model
SAVED_MODEL_PATH = "results/gemma-2b-lora-mac-simple"
BASE_MODEL = "google/gemma-2b-it"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",
    torch_dtype=torch.float32
)

# Load your fine-tuned LoRA adapter on top of the base model
model = PeftModel.from_pretrained(base_model, SAVED_MODEL_PATH)

# Function to generate sentiment predictions


def predict_sentiment(text):
    # Prepare input text
    input_text = f"Sentiment: {text}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print results
    print(f"Input: {text}")
    print(f"Full output: {generated_text}")

    # Extract just the answer part
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[1].strip()
        print(f"Prediction: {answer}")


# Test with some examples
test_examples = [
    "This movie was absolutely fantastic!",
    "I was very disappointed with the terrible service.",
    "The product works as expected, nothing special.",
    "I love this book, couldn't put it down!",
    "The restaurant was overpriced and the food was mediocre."
]

print("Testing your fine-tuned Gemma 2B model...\n")
for example in test_examples:
    predict_sentiment(example)
    print("-" * 50)
