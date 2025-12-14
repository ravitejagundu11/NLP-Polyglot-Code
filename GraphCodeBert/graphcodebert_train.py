"""
GraphCodeBERT Training Script for SemEval-2026 Task 13 Subtask A
AI-Generated Code Detection - Standalone GPU-ready Script

Usage:
    python graphcodebert_train.py --train_file ./data/train.csv --test_file ./data/test.csv

Author: SemEval-2026 Task 13
"""

import os
import re
import random
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset  # Hugging Face datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model
    MODEL_NAME = "microsoft/graphcodebert-base"
    
    # Training
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 2
    
    # Regularization
    DROPOUT_RATE = 0.1
    LABEL_SMOOTHING = 0.1
    
    # Features
    USE_ADDITIONAL_FEATURES = True
    NUM_ADDITIONAL_FEATURES = 40
    
    # Paths
    OUTPUT_DIR = "./outputs"
    MODEL_SAVE_DIR = "./saved_models"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reproducibility
    SEED = 42
    
    # Cross-validation
    USE_CV = True
    N_FOLDS = 5


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()


# FEATURE EXTRACTION (Language-Agnostic)

def extract_code_features(code: str) -> List[float]:
    """
    Extract language-agnostic features from code
    These help generalize to unseen programming languages
    """
    features = []
    
    # Basic statistics
    code_length = len(code)
    num_lines = code.count('\n') + 1
    avg_line_length = code_length / max(num_lines, 1)
    
    features.extend([code_length, num_lines, avg_line_length])
    
    # Character-level features
    num_spaces = code.count(' ')
    num_tabs = code.count('\t')
    num_newlines = code.count('\n')
    
    features.extend([num_spaces, num_tabs, num_newlines])
    
    # Punctuation and operators
    num_braces = code.count('{') + code.count('}')
    num_brackets = code.count('[') + code.count(']')
    num_parens = code.count('(') + code.count(')')
    num_semicolons = code.count(';')
    num_commas = code.count(',')
    num_dots = code.count('.')
    num_colons = code.count(':')
    
    features.extend([num_braces, num_brackets, num_parens, 
                     num_semicolons, num_commas, num_dots, num_colons])
    
    # Mathematical operators
    num_operators = (code.count('+') + code.count('-') + 
                    code.count('*') + code.count('/') +
                    code.count('=') + code.count('<') + 
                    code.count('>') + code.count('%'))
    
    features.append(num_operators)
    
    # Word-level features
    words = re.findall(r'\b\w+\b', code)
    num_words = len(words)
    unique_words = len(set(words))
    word_diversity = unique_words / max(num_words, 1)
    
    features.extend([num_words, unique_words, word_diversity])
    
    # Token length statistics
    if words:
        token_lengths = [len(w) for w in words]
        avg_token_len = np.mean(token_lengths)
        std_token_len = np.std(token_lengths)
        max_token_len = max(token_lengths)
    else:
        avg_token_len = std_token_len = max_token_len = 0
    
    features.extend([avg_token_len, std_token_len, max_token_len])
    
    # Comment indicators
    num_single_comments = code.count('//') + code.count('#')
    num_multi_comments = code.count('/*') + code.count('"""') + code.count("'''")
    
    features.extend([num_single_comments, num_multi_comments])
    
    # Indentation patterns
    lines = code.split('\n')
    indents = []
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            indents.append(indent)
    
    if indents:
        avg_indent = np.mean(indents)
        std_indent = np.std(indents)
        max_indent = max(indents)
    else:
        avg_indent = std_indent = max_indent = 0
    
    features.extend([avg_indent, std_indent, max_indent])
    
    # Character entropy
    char_counts = {}
    for char in code:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    if code_length > 0:
        char_probs = [count / code_length for count in char_counts.values()]
        char_entropy = -sum(p * np.log2(p) for p in char_probs if p > 0)
    else:
        char_entropy = 0
    
    features.append(char_entropy)
    
    # Token entropy
    if words:
        token_counts = {}
        for token in words:
            token_counts[token] = token_counts.get(token, 0) + 1
        token_probs = [count / len(words) for count in token_counts.values()]
        token_entropy = -sum(p * np.log2(p) for p in token_probs if p > 0)
    else:
        token_entropy = 0
    
    features.append(token_entropy)
    
    # Line length statistics
    non_empty_lines = [line for line in lines if line.strip()]
    if non_empty_lines:
        line_lengths = [len(line) for line in non_empty_lines]
        line_len_mean = np.mean(line_lengths)
        line_len_std = np.std(line_lengths)
        line_len_max = max(line_lengths)
        line_len_min = min(line_lengths)
    else:
        line_len_mean = line_len_std = line_len_max = line_len_min = 0
    
    features.extend([line_len_mean, line_len_std, line_len_max, line_len_min])
    
    # Complexity approximation
    complexity = (code.count('if') + code.count('for') + 
                  code.count('while') + code.count('case') +
                  code.count('catch') + code.count('&&') + 
                  code.count('||') + code.count('?'))
    
    features.append(complexity)
    
    # Ratios (normalized features)
    if code_length > 0:
        space_ratio = num_spaces / code_length
        operator_ratio = num_operators / code_length
        punct_ratio = (num_braces + num_brackets + num_parens) / code_length
    else:
        space_ratio = operator_ratio = punct_ratio = 0
    
    features.extend([space_ratio, operator_ratio, punct_ratio])
    
    # Pad to ensure consistent feature count
    while len(features) < Config.NUM_ADDITIONAL_FEATURES:
        features.append(0.0)
    
    return features[:Config.NUM_ADDITIONAL_FEATURES]


# DATASET

class CodeDataset(Dataset):
    """Dataset for code classification"""
    
    def __init__(
        self,
        codes: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512,
        use_features: bool = True
    ):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_features = use_features
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx: int) -> Dict:
        code = self.codes[idx]
        label = self.labels[idx]
        
        # Handle None/NaN values
        if code is None or (isinstance(code, float) and np.isnan(code)):
            code = ""
        code = str(code)
        
        # Tokenize
        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(int(label), dtype=torch.long)
        }
        
        # Extract features
        if self.use_features:
            features = extract_code_features(code)
            item['features'] = torch.tensor(features, dtype=torch.float32)
        
        return item


# MODEL

class GraphCodeBERTClassifier(nn.Module):
    """
    GraphCodeBERT with additional feature integration
    for AI-generated code detection
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/graphcodebert-base",
        num_labels: int = 2,
        use_features: bool = True,
        num_features: int = 40,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.use_features = use_features
        
        # Load GraphCodeBERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size  # 768 for base
        
        # Feature processing MLP
        if use_features:
            self.feature_mlp = nn.Sequential(
                nn.Linear(num_features, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            combined_size = hidden_size + 64
        else:
            combined_size = hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        
        # Combine with features
        if self.use_features and features is not None:
            feature_emb = self.feature_mlp(features)
            combined = torch.cat([pooled, feature_emb], dim=1)
        else:
            combined = pooled
        
        # Classify
        logits = self.classifier(combined)
        
        return logits


# TRAINING

def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    progress = tqdm(dataloader, desc="Training", leave=False)
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            features = batch.get('features')
            if features is not None:
                features = features.to(device)
            
            # Forward
            logits = model(input_ids, attention_mask, features)
            loss = criterion(logits, labels)
            
            # Backward with gradient accumulation
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
            num_batches += 1
            progress.set_postfix({'loss': f'{total_loss / num_batches:.4f}'})
            
        except Exception as e:
            print(f"\nWarning: Error in batch {step}: {e}")
            optimizer.zero_grad()  # Reset gradients on error
            continue
    
    # Handle remaining gradients (if total steps not divisible by accumulation steps)
    if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            features = batch.get('features')
            if features is not None:
                features = features.to(device)
            
            logits = model(input_ids, attention_mask, features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Metrics
    metrics = {
        'loss': total_loss / len(dataloader),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro')
    }
    
    return metrics, all_preds, all_probs


def train_model(
    train_codes, train_labels,
    val_codes, val_labels,
    config, fold=0
):
    """Train a single model"""
    
    print(f"\n{'='*60}")
    print(f"Training GraphCodeBERT - Fold {fold}")
    print(f"Train samples: {len(train_codes)}, Val samples: {len(val_codes)}")
    print(f"Device: {config.DEVICE}")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create datasets
    train_dataset = CodeDataset(
        train_codes, train_labels, tokenizer,
        max_length=config.MAX_LENGTH,
        use_features=config.USE_ADDITIONAL_FEATURES
    )
    
    val_dataset = CodeDataset(
        val_codes, val_labels, tokenizer,
        max_length=config.MAX_LENGTH,
        use_features=config.USE_ADDITIONAL_FEATURES
    )
    
    # Dataloaders (num_workers=0 for compatibility, increase if no issues)
    num_workers = 0 if config.DEVICE == "cpu" else 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(config.DEVICE == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(config.DEVICE == "cuda")
    )
    
    # Initialize model
    model = GraphCodeBERTClassifier(
        model_name=config.MODEL_NAME,
        num_labels=2,
        use_features=config.USE_ADDITIONAL_FEATURES,
        num_features=config.NUM_ADDITIONAL_FEATURES,
        dropout=config.DROPOUT_RATE
    )
    model = model.to(config.DEVICE)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    total_steps = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop with early stopping
    best_f1 = 0
    best_model_state = None
    patience = 2  # Early stopping patience
    no_improve_count = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            config.DEVICE, config
        )
        
        # Evaluate
        val_metrics, _, _ = evaluate(model, val_loader, config.DEVICE)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        
        # Save best
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
            print(f"★ New best F1: {best_f1:.4f}")
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epoch(s)")
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save best model
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_SAVE_DIR, f"graphcodebert_fold{fold}_best.pt")
    
    if best_model_state is not None:
        torch.save(best_model_state, model_path)
        print(f"\nSaved best model to: {model_path}")
    else:
        # Save current model if no best was found
        torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, model_path)
        print(f"\nSaved final model to: {model_path}")
    
    # Clear GPU memory after each fold
    del model
    clear_gpu_memory()
    
    return model_path, best_f1, tokenizer


# INFERENCE

def predict(model, tokenizer, test_codes, config):
    """Make predictions on test data"""
    
    model.eval()
    
    # Create dataset (dummy labels)
    test_dataset = CodeDataset(
        test_codes, [0] * len(test_codes), tokenizer,
        max_length=config.MAX_LENGTH,
        use_features=config.USE_ADDITIONAL_FEATURES
    )
    
    num_workers = 0 if config.DEVICE == "cpu" else 2
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(config.DEVICE == "cuda")
    )
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            features = batch.get('features')
            if features is not None:
                features = features.to(config.DEVICE)
            
            logits = model(input_ids, attention_mask, features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return all_preds, all_probs


def ensemble_predict(model_paths, tokenizer, test_codes, config):
    """Ensemble predictions from multiple models"""
    
    all_probs = []
    
    for model_path in model_paths:
        print(f"Loading: {model_path}")
        
        # Load model
        model = GraphCodeBERTClassifier(
            model_name=config.MODEL_NAME,
            num_labels=2,
            use_features=config.USE_ADDITIONAL_FEATURES,
            num_features=config.NUM_ADDITIONAL_FEATURES,
            dropout=config.DROPOUT_RATE
        )
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model = model.to(config.DEVICE)
        
        # Predict
        _, probs = predict(model, tokenizer, test_codes, config)
        all_probs.append(np.array(probs))
        
        # Clear memory after each model
        del model
        clear_gpu_memory()
    
    # Average probabilities
    avg_probs = np.mean(all_probs, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)
    
    return final_preds, avg_probs


# MAIN
def main():
    # Hugging Face Dataset Settings
    HF_DATASET_NAME = "DaniilOr/SemEval-2026-Task13"  # Hugging Face dataset name
    CODE_COLUMN = "code"                     # Column name for code in the dataset
    LABEL_COLUMN = "label"                   # Column name for labels in the dataset
    
    OUTPUT_FILE = "./outputs/predictions.csv" # Path to save predictions
    
    # Training parameters
    BATCH_SIZE = 128                         # Batch size (reduce if OOM error)
    NUM_EPOCHS = 5                           # Number of training epochs
    LEARNING_RATE = 2e-5                     # Learning rate
    MAX_LENGTH = 512                         # Max sequence length for tokenizer
    
    # Cross-validation settings
    USE_CROSS_VALIDATION = True              # Set to False for simple train/val split
    N_FOLDS = 5                              # Number of CV folds
    
    # Other settings
    SEED = 42                                # Random seed for reproducibility

    
    # Update config with parameters
    Config.BATCH_SIZE = BATCH_SIZE
    Config.NUM_EPOCHS = NUM_EPOCHS
    Config.LEARNING_RATE = LEARNING_RATE
    Config.MAX_LENGTH = MAX_LENGTH
    Config.N_FOLDS = N_FOLDS
    Config.USE_CV = USE_CROSS_VALIDATION
    Config.SEED = SEED
    
    # Set seed
    set_seed(Config.SEED)
    
    print("\n" + "="*60)
    print("GraphCodeBERT - AI-Generated Code Detection")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Max Length: {Config.MAX_LENGTH}")
    print(f"Use CV: {Config.USE_CV}")
    print(f"N Folds: {Config.N_FOLDS}")
    print("="*60 + "\n")
    
    # LOAD DATA (from Hugging Face)
    
    print(f"Loading dataset from Hugging Face: {HF_DATASET_NAME}")
    print("Downloading dataset... (this may take a moment)")
    
    try:
        dataset = load_dataset(HF_DATASET_NAME, 'A')
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Get train split
        if 'train' in dataset:
            train_data = dataset['train']
        else:
            # Use first available split
            first_split = list(dataset.keys())[0]
            train_data = dataset[first_split]
            print(f"Using '{first_split}' split as training data")
        
        # Convert to lists
        codes = train_data[CODE_COLUMN]
        labels = train_data[LABEL_COLUMN]
        
        # Convert to Python lists if needed
        if hasattr(codes, 'tolist'):
            codes = codes.tolist()
        else:
            codes = list(codes)
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        else:
            labels = list(labels)
        
        # Validate data
        assert len(codes) == len(labels), f"Mismatch: {len(codes)} codes vs {len(labels)} labels"
        assert len(codes) > 0, "No training samples found!"
        
        # Check for None values
        none_count = sum(1 for c in codes if c is None)
        if none_count > 0:
            print(f"Warning: Found {none_count} None values in code column")
        
        print(f"Loaded {len(codes)} training samples from Hugging Face")
        
        # Load test split if available
        test_codes = None
        test_labels = None
        if 'test' in dataset:
            test_data = dataset['test']
            test_codes = test_data[CODE_COLUMN]
            if hasattr(test_codes, 'tolist'):
                test_codes = test_codes.tolist()
            else:
                test_codes = list(test_codes)
            if LABEL_COLUMN in test_data.features:
                test_labels = test_data[LABEL_COLUMN]
                if hasattr(test_labels, 'tolist'):
                    test_labels = test_labels.tolist()
                else:
                    test_labels = list(test_labels)
            print(f"Loaded {len(test_codes)} test samples from Hugging Face")
        
    except Exception as e:
        print(f"Error loading from Hugging Face: {e}")
        print("Please check the dataset name.")
        return
    
    # Print label distribution
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    model_paths = []
    fold_results = []
    tokenizer = None
    
    if Config.USE_CV:
        # Cross-validation
        kfold = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(codes, labels)):
            train_codes = [codes[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_codes = [codes[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            model_path, best_f1, tokenizer = train_model(
                train_codes, train_labels,
                val_codes, val_labels,
                Config, fold=fold
            )
            
            model_paths.append(model_path)
            fold_results.append({'fold': fold, 'f1': best_f1})
        
        # Print CV summary
        print("\n" + "="*60)
        print("Cross-Validation Summary")
        print("="*60)
        for r in fold_results:
            print(f"Fold {r['fold']}: F1 = {r['f1']:.4f}")
        
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        std_f1 = np.std([r['f1'] for r in fold_results])
        print(f"\nMean F1: {avg_f1:.4f} ± {std_f1:.4f}")
        print("="*60)
    
    else:
        # Simple train/val split
        train_codes, val_codes, train_labels, val_labels = train_test_split(
            codes, labels, test_size=0.2, random_state=Config.SEED, stratify=labels
        )
        
        model_path, best_f1, tokenizer = train_model(
            train_codes, train_labels,
            val_codes, val_labels,
            Config, fold=0
        )
        model_paths.append(model_path)
    
    # Test set prediction
    if test_codes is not None and len(test_codes) > 0:
        print("\n" + "="*60)
        print("Making predictions on test set...")
        print("="*60)
        print(f"Test samples: {len(test_codes)}")
        
        if len(model_paths) > 1:
            # Ensemble prediction
            print(f"Using ensemble of {len(model_paths)} models")
            predictions, probabilities = ensemble_predict(
                model_paths, tokenizer, test_codes, Config
            )
        else:
            # Single model prediction
            model = GraphCodeBERTClassifier(
                model_name=Config.MODEL_NAME,
                num_labels=2,
                use_features=Config.USE_ADDITIONAL_FEATURES,
                num_features=Config.NUM_ADDITIONAL_FEATURES,
                dropout=Config.DROPOUT_RATE
            )
            model.load_state_dict(torch.load(model_paths[0], map_location=Config.DEVICE))
            model = model.to(Config.DEVICE)
            
            predictions, probabilities = predict(model, tokenizer, test_codes, Config)
        
        # Save predictions
        os.makedirs(os.path.dirname(OUTPUT_FILE) if os.path.dirname(OUTPUT_FILE) else '.', exist_ok=True)
        
        results_df = pd.DataFrame({
            'prediction': predictions,
            'prob_human': [p[0] for p in probabilities],
            'prob_ai': [p[1] for p in probabilities]
        })
        
        results_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nPredictions saved to: {OUTPUT_FILE}")
        
        # If test labels available, compute metrics
        if test_labels is not None:
            test_f1 = f1_score(test_labels, predictions, average='macro')
            test_acc = accuracy_score(test_labels, predictions)
            
            print(f"\nTest Results:")
            print(f"F1 Score: {test_f1:.4f}")
            print(f"Accuracy: {test_acc:.4f}")
            print("\nClassification Report:")
            print(classification_report(test_labels, predictions, 
                                       target_names=['Human', 'AI-Generated']))
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Saved models: {model_paths}")
    print(f"Output directory: {Config.MODEL_SAVE_DIR}")


if __name__ == "__main__":
    main()
