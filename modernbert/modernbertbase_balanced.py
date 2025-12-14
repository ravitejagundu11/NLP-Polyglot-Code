import pandas as pd
import torch
import logging
import sys
import os
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
# PATHS TO YOUR SAVED FILES
TRAIN_FILE = "/scratch/gilbreth/navad01/modernbertbase_balanced/balanced_train.csv"
VAL_FILE   = "/scratch/gilbreth/navad01/modernbertbase_balanced/downsized_validation.csv"

# Directories
BASE_DIR = "/scratch/gilbreth/navad01/modernbertbase_balanced"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "final_model")
LOG_FILE = os.path.join(BASE_DIR, "final_training_log.txt")

MODEL_NAME = "answerdotai/ModernBERT-base"

# ModernBERT context is HUGE (8192)
MAX_LEN = 8192

# --- A100 OPTIMIZATIONS FOR 8K CONTEXT ---
# We cannot use 128 batch size here. 
# 16 is safe for 80GB VRAM with Gradient Checkpointing enabled.
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16

# ==========================================
# 2. SETUP LOGGING & DIRECTORIES
# ==========================================
os.makedirs(BASE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 3. DATASET CLASS
# ==========================================
class CodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=8192):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = str(self.texts[idx])
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                padding=False, 
                return_tensors=None
            )
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'label': self.labels[idx]
            }
        except Exception as e:
            return None

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average='weighted')
    f1_m = f1_score(labels, preds, average='macro') 
    
    return {'accuracy': acc, 'f1_weighted': f1_w, 'f1_macro': f1_m}

# ==========================================
# 4. MAIN TRAINING
# ==========================================
def main():
    logger.info(f"--- Starting A100 OPTIMIZED Training Job for ModernBERT ---")

    # 1. Load Pre-Prepared Data
    logger.info(f"Loading training data from {TRAIN_FILE}...")
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        df_val = pd.read_csv(VAL_FILE)
        logger.info(f"Train Size: {len(df_train)}")
        logger.info(f"Val Size:   {len(df_val)}")
    except FileNotFoundError as e:
        logger.error(f"Could not find CSV files! Run prepare_data.py first. Error: {e}")
        return

    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )

    # 3. Create Datasets
    train_dataset = CodeDataset(df_train['code'].tolist(), df_train['label'].tolist(), tokenizer, MAX_LEN)
    val_dataset = CodeDataset(df_val['code'].tolist(), df_val['label'].tolist(), tokenizer, MAX_LEN)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        bf16=True,                  
        dataloader_num_workers=16,     
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # 5. Train & Save
    logger.info(f"Starting Training (Batch Size: {TRAIN_BATCH_SIZE} | Context: {MAX_LEN})...")
    trainer.train()
    
    logger.info(f"Saving Best Model to {FINAL_MODEL_DIR}...")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    logger.info("Saving explicit .pt file...")
    torch.save(model.state_dict(), os.path.join(FINAL_MODEL_DIR, "model.pt"))
    logger.info(f"Saved model.pt to {FINAL_MODEL_DIR}")

    # ==========================================
    # 6. TEST SET EVALUATION
    # ==========================================
    logger.info("--- Starting Evaluation on TEST Set ---")
    
    try:
        ds_test = load_dataset("DaniilOr/SemEval-2026-Task13", "A", split="test")
        df_test = ds_test.to_pandas()
        if 'text' in df_test.columns: df_test = df_test.rename(columns={'text': 'code'})
        
        logger.info(f"Test Set Size: {len(df_test)}")

        test_dataset = CodeDataset(
            df_test['code'].tolist(), 
            df_test['label'].tolist(), 
            tokenizer, 
            MAX_LEN
        )

        test_results = trainer.predict(test_dataset)
        
        logger.info("xxx FINAL TEST RESULTS xxx")
        logger.info(f"Test Accuracy: {test_results.metrics['test_accuracy']:.4f}")
        logger.info(f"Test Macro F1: {test_results.metrics['test_f1_macro']:.4f}")
        logger.info("xxxxxxxxxxxxxxxxxxxxxxxxxx")

        # Save predictions
        predictions = test_results.predictions.argmax(-1)
        df_test['predicted_label'] = predictions
        prediction_file = os.path.join(OUTPUT_DIR, "test_predictions.csv")
        df_test.to_csv(prediction_file, index=False)
        logger.info(f"Predictions saved to {prediction_file}")

    except Exception as e:
        logger.warning(f"Could not run test set evaluation: {e}")

    logger.info("Job Complete.")

if __name__ == "__main__":
    main()