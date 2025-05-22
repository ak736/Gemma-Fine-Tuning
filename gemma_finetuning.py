"""
Simplified Gemma 2B Fine-Tuning for MacBook Air M2

This script provides an even more simplified version for fine-tuning Gemma 2B on Mac.
It uses direct supervised fine-tuning approach for a sentiment analysis task.
"""

import os
import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "google/gemma-2b-it"
OUTPUT_DIR = "results/gemma-2b-lora-mac-simple"
NUM_TRAIN_SAMPLES = 100  # Extremely small sample for quick training
NUM_VAL_SAMPLES = 20
MAX_LENGTH = 64  # Reduced sequence length
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
EPOCHS = 2

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FORCE CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Disable MPS (Metal) backend in PyTorch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Fall back to CPU
torch.set_num_threads(2)  # Limit threads to prevent overheating

def get_dataset():
    """Load a small subset of SST2 dataset."""
    logger.info("Loading dataset...")
    dataset = load_dataset("glue", "sst2")
    
    # Take very small subsets
    train_dataset = dataset["train"].shuffle(seed=42).select(range(NUM_TRAIN_SAMPLES))
    val_dataset = dataset["validation"].shuffle(seed=42).select(range(NUM_VAL_SAMPLES))
    
    return train_dataset, val_dataset

def prepare_dataset(dataset, tokenizer):
    """Prepare dataset for training."""
    
    def preprocess_function(examples):
        # Create simple templates for sentiment analysis
        inputs = []
        for sentence in examples["sentence"]:
            # Create a simple prompt - kept very short
            inputs.append(f"Sentiment: {sentence}\nAnswer:")
        
        # Create targets/completions
        targets = []
        for label in examples["label"]:
            targets.append(" positive" if label == 1 else " negative")
        
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs, 
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True
        )
        
        # Tokenize targets
        labels = tokenizer(
            targets,
            max_length=8,  # Short targets
            padding="max_length", 
            truncation=True
        )
        
        # Prepare the labels - set padding to -100 to be ignored in loss
        for i in range(len(labels["input_ids"])):
            # Set padding token labels to -100
            labels["input_ids"][i] = [
                label if label != tokenizer.pad_token_id else -100
                for label in labels["input_ids"][i]
            ]
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=4,  # Process in very small batches
        remove_columns=dataset.column_names,
        desc="Processing dataset",
    )
    
    return processed_dataset

def main():
    logger.info(f"Starting fine-tuning Gemma 2B on Mac")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Make sure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_dataset, val_dataset = get_dataset()
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_dataset, tokenizer)
    val_dataset = prepare_dataset(val_dataset, tokenizer)
    
    # Load model - FORCE CPU
    logger.info("Loading model on CPU...")
    
    # Explicitly tell PyTorch to use CPU
    torch_device = torch.device("cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use FP32 for CPU
        low_cpu_mem_usage=True,
    ).to(torch_device)  # Explicitly move to CPU
    
    # Configure LoRA with minimal target modules
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=4,  # Reduced even further
        lora_alpha=8,  # Reduced alpha
        lora_dropout=0.0,  # Remove dropout to save compute
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj"],  # Target only q projection
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params} ({trainable_params/all_params:.2%})")
    
    # Configure minimal training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=1e-4,
        weight_decay=0.0,  # Reduce computation
        logging_steps=5,
        eval_steps=20,
        save_steps=50,
        save_total_limit=1,
        # Explicitly disable GPU usage
        no_cuda=True,
        # Disable reporting to save memory
        report_to="none",
        # Simplify training
        remove_unused_columns=False,
        disable_tqdm=False,
    )
    
    # Initialize Trainer with explicit CPU device
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Explicitly check we're on CPU
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving fine-tuned model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info(f"Fine-tuning complete! Model saved to {OUTPUT_DIR}")
    
    # Test model with a sample
    logger.info("Testing inference with a sample...")
    test_text = "Sentiment: This movie was absolutely fantastic!\nAnswer:"
    inputs = tokenizer(test_text, return_tensors="pt").to(torch_device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Input: {test_text}")
    logger.info(f"Output: {generated_text}")

if __name__ == "__main__":
    main()
