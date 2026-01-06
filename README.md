# Polyglot Code Forensics: Detecting LLM-Generated Code

A comprehensive research project for detecting AI-generated code using state-of-the-art transformer models. This repository implements multiple code understanding models for binary classification of human-written vs. LLM-generated code, developed for **SemEval-2026 Task 13 Subtask A**.

> **For detailed methodology, experiments, and results, please refer to our [research paper](documents/nlpResearchPaper_final.pdf).**

## Project Overview

This project tackles the challenge of distinguishing between human-written and AI-generated code across multiple programming languages. We evaluate various pre-trained models specialized in code understanding, including:

- **CodeBERT** - Microsoft's bimodal pre-trained model for programming and natural languages
- **GraphCodeBERT** - Code understanding model with data flow awareness
- **UniXcoder** - Unified cross-modal pre-trained model for code representation
- **ModernBERT** - State-of-the-art BERT variant optimized for long contexts
- **StarCoder** - Large language model series (3B and 7B parameters) for code generation and understanding
- **CatBoost** - Gradient boosting baseline for comparison

## Repository Structure

```
Polyglot-Code/
├── Catboost/              # CatBoost gradient boosting implementation
├── Code-bert/             # CodeBERT model training scripts
├── GraphCodeBert/         # GraphCodeBERT implementation
├── modernbert/            # ModernBERT model scripts
├── StarCoder/             # StarCoder model variants (3B & 7B)
├── unxicoder/             # UniXcoder model implementation
├── random-forest/         # Random forest baseline
└── documents/             # Documentation and resources
```

## Getting Started

### Prerequisites

```bash
# Core dependencies
pip install torch transformers datasets accelerate
pip install evaluate seqeval scikit-learn pandas numpy
pip install peft bitsandbytes  # For parameter-efficient fine-tuning
```

### Dataset

The project uses the **SemEval-2026 Task 13 Dataset** from Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("DaniilOr/SemEval-2026-Task13", 'A')
```

## Models & Implementation

### CodeBERT
- **Model**: `microsoft/codebert-base`
- **Scripts**: [CODE-BERT-UPDATED.py](Code-bert/CODE-BERT-UPDATED.py), [CODE-BERT.ipynb](Code-bert/CODE-BERT.ipynb)
- **Features**: Bimodal understanding of code and natural language

### GraphCodeBERT
- **Model**: `microsoft/graphcodebert-base`
- **Scripts**: [graphcodebert_train.py](GraphCodeBert/graphcodebert_train.py), [graphcoderbert_balanced.py](GraphCodeBert/graphcoderbert_balanced.py)
- **Max Length**: 512 tokens
- **Batch Size**: 16
- **Features**: Data flow-aware code representation

### StarCoder
- **Models**: 
  - `bigcode/starcoder2-3b` (3B parameters)
  - `bigcode/starcoder2-7b` (7B parameters)
- **Scripts**: [starcoder2-3b.py](StarCoder/starcoder2-3b.py), [starcoder-7b.py](StarCoder/starcoder-7b.py)
- **Max Length**: 8192 tokens
- **Features**: Large-scale code generation and understanding

### ModernBERT
- **Model**: `answerdotai/ModernBERT-base`
- **Scripts**: [modernbert.py](modernbert/modernbert.py), [modernbertbase_balanced.py](modernbert/modernbertbase_balanced.py)
- **Max Length**: 8192 tokens
- **Features**: Optimized for long-context understanding

### UniXcoder
- **Model**: `microsoft/unixcoder-base`
- **Scripts**: [unxicoder.py](unxicoder/unxicoder.py), [unxicoder_balanced.py](unxicoder/unxicoder_balanced.py)
- **Max Length**: 512 tokens
- **Features**: Unified cross-modal code representation

## Training

Each model directory contains training scripts with the following general structure:

```python
# Example training configuration
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
MAX_LENGTH = 512  # or 8192 for ModernBERT/StarCoder
```

### Running Training

```bash
# Example: GraphCodeBERT
python GraphCodeBert/graphcodebert_train.py --train_file ./data/train.csv --test_file ./data/test.csv

# Example: StarCoder 7B
python StarCoder/starcoder-7b.py

# Example: ModernBERT
python modernbert/modernbert.py
```

## Evaluation Metrics

The models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positive rate
- **Recall**: Sensitivity measure

## Key Features

- Multiple state-of-the-art code understanding models
- Support for balanced and imbalanced datasets
- Parameter-efficient fine-tuning with PEFT
- GPU-optimized training scripts
- Comprehensive evaluation metrics
- Support for long sequences (up to 8192 tokens)
- Cross-lingual code detection

## License

See [LICENSE](LICENSE) file for details.

## Contributing

This is a research project for SemEval-2026 Task 13. For questions or collaborations, please open an issue.

## Related Resources

- [SemEval-2026 Task 13 Dataset](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13)
- [CodeBERT](https://huggingface.co/microsoft/codebert-base)
- [GraphCodeBERT](https://huggingface.co/microsoft/graphcodebert-base)
- [StarCoder](https://huggingface.co/bigcode)
- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
- [UniXcoder](https://huggingface.co/microsoft/unixcoder-base)