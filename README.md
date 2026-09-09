## 💻 Usage & How To Run

### 🇬🇧 English Language Focus
> ⚠️ **Note on Training Data:** While the pipeline relies on a multilingual dataset framework, this specific instance of `abusive_model.pth` has been strictly **trained and optimized using English text data**. It is ready for deployment on English text classifications.

### 🌐 Extending to Other Languages
The underlying data pipeline supports a variety of global languages (such as Hindi, Arabic, Russian, and Spanish). If you want to train this architecture to recognize abusive words in another language, you can modify the dataset configuration inside `data.py` to point to a different language locale before executing the training loop.

---

### Run Instructions

#### 1. Prepare and Verify Data Pipeline
Downloads or slices the required subsets from the raw dataset:
```bash
python data.py
```

#### 2. Model Training & Fine-Tuning
Orchestrates the neural network pipeline using PyTorch to output the weights file:
```bash
python main.py
```

#### 3. Real-Time Inference Testing
To test sentences locally using your trained `abusive_model.pth`, run the evaluation module:
```bash
python model_testing.py
```

#### 📋 Quick Inference Example (Python)
You can test phrases directly in your script like this:
```python
import torch
# Example conceptual inference check
text_to_test = "Your sample text here"
# The model will predict if it falls into the toxic/abusive distribution
```
