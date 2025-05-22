# Fine-Tuning Gemma 2B 🚀

A practical implementation of fine-tuning Google's Gemma 2B model for sentiment analysis on Apple Silicon hardware using parameter-efficient methods.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📖 Overview

This project demonstrates how to fine-tune Google's Gemma 2B instruction-tuned model on resource-constrained hardware (MacBook Air M2 with 8GB RAM). Using LoRA (Low-Rank Adaptation) and aggressive memory optimization techniques, we successfully fine-tuned the model for sentiment analysis tasks.

### 🎯 Key Features

- **Memory-Efficient**: Optimized for Apple Silicon Macs with limited RAM
- **LoRA Fine-Tuning**: Parameter-efficient training using Low-Rank Adaptation
- **CPU-Only Training**: Works without requiring expensive GPU hardware
- **Educational Focus**: Designed for learning and understanding the fine-tuning process

## 🔧 Technical Specifications

| Component | Details |
|-----------|---------|
| **Base Model** | `google/gemma-2b-it` (2.5B parameters) |
| **Method** | LoRA (Low-Rank Adaptation) |
| **Hardware** | MacBook Air M2, 8GB RAM |
| **Training Time** | ~4 hours for 100 samples, 2 epochs |
| **Memory Usage** | ~6-8GB RAM during training |
| **Trainable Parameters** | 921,600 (0.04% of total model) |

## 📊 Results

Due to computational constraints, we trained on a limited dataset:

- **Training Samples**: 100 examples from SST-2 dataset
- **Validation Samples**: 20 examples
- **Training Epochs**: 2
- **Max Sequence Length**: 64 tokens

### Sample Outputs

```
Input: "This movie was absolutely fantastic!"
Output: "Thank you for your feedback! I'm glad..."

Input: "I was very disappointed with the terrible service."
Output: "I'm sorry to hear that you had a..."
```

**Note**: The model learned to respond conversationally to sentiment rather than producing classification labels. This demonstrates successful fine-tuning, though the task format could be improved with more training data and clearer instructions.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MacBook with Apple Silicon (M1/M2/M3) or Intel Mac
- At least 8GB RAM (16GB recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/gemma-2b-finetuning.git
   cd gemma-2b-finetuning
   ```

2. **Create virtual environment**
   ```bash
   python -m venv gemma-env
   source gemma-env/bin/activate  # On Windows: gemma-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Authenticate with Hugging Face**
   ```bash
   huggingface-cli login
   ```
   You'll need to accept the Gemma license at [https://huggingface.co/google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)

### Training

```bash
python simple_finetuning.py
```

The script will:
- Download the Gemma 2B model (~5GB)
- Load a subset of the SST-2 sentiment analysis dataset
- Fine-tune using LoRA for 2 epochs
- Save the trained model to `results/gemma-2b-lora-mac-simple`

### Testing the Model

After training, test your model:

```bash
python test_model.py
```

## 💡 Usage Examples

### Loading Your Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load the fine-tuned model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it", 
    device_map="cpu"
)
model = PeftModel.from_pretrained(base_model, "results/gemma-2b-lora-mac-simple")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

# Generate predictions
def predict_sentiment(text):
    input_text = f"Sentiment: {text}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
result = predict_sentiment("This product exceeded my expectations!")
print(result)
```

## 📈 Improving Results

The current model was trained with minimal resources to demonstrate feasibility. For better performance:

### 🔧 Increase Training Data
```python
NUM_TRAIN_SAMPLES = 1000  # Instead of 100
NUM_VAL_SAMPLES = 200     # Instead of 20
```

### ⏰ Train Longer
```python
EPOCHS = 5               # Instead of 2
MAX_LENGTH = 128         # Instead of 64
```

### 🎯 Better LoRA Configuration
```python
lora_config = LoraConfig(
    r=16,                    # Instead of 4
    lora_alpha=32,           # Instead of 8
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # More modules
)
```

### 📝 Improved Prompting
```python
# More explicit task formatting
prompt = f"Task: Classify sentiment as POSITIVE or NEGATIVE.\nText: {text}\nSentiment:"
target = "POSITIVE" if label == 1 else "NEGATIVE"
```

## 🔧 Hardware Considerations


### Performance Tips

- **Close unnecessary applications** during training
- **Connect to power** to prevent thermal throttling
- **Use `caffeinate -i`** to prevent system sleep
- **Monitor memory usage** with Activity Monitor

## 📁 Project Structure

```
gemma-2b-finetuning/
├── gemma_finetuning.py      # Main training script
├── test_model.py             # Model testing script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── results/                  # Output directory
    └── gemma-2b-lora-mac-simple/
        ├── adapter_config.json
        ├── adapter_model.bin
        └── ...
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Some areas for improvement:

- [ ] Better evaluation metrics
- [ ] Support for other tasks (summarization, QA)
- [ ] Quantization optimizations
- [ ] Web interface for testing
- [ ] Batch inference support

## 📚 Resources

- [Gemma Model Card](https://huggingface.co/google/gemma-2b-it)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## ⚠️ Limitations

- **Small training dataset**: Only 100 examples due to hardware constraints
- **Limited epochs**: 2 epochs may not be sufficient for optimal convergence
- **Conversational outputs**: Model produces chat-like responses rather than classification labels
- **Hardware dependent**: Optimized specifically for Apple Silicon Macs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google for open-sourcing the Gemma model family
- Hugging Face for the transformers and PEFT libraries
- The open-source community for making AI accessible

## 📞 Contact

Feel free to reach out if you have questions or suggestions!

- GitHub: [@ak736](https://github.com/ak736)
- Email: aniketkir63@gmail.com

---

**⭐ If this project helped you, please give it a star!**

> "The best way to learn is by doing. This project proves that even with limited resources, you can successfully fine-tune state-of-the-art language models."
