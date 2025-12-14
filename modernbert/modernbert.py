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
HF_DATASET_ID = "DaniilOr/SemEval-2026-Task13"
HF_SUBSET = "A"

# Consolidated Directory
BASE_DIR = "/scratch/gilbreth/navad01/modernbert"
OUTPUT_DIR = BASE_DIR 
FINAL_MODEL_DIR = BASE_DIR   
LOG_FILE = os.path.join(BASE_DIR, "final_training_log.txt") 

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LEN = 8192
BATCH_SIZE = 16 

# ==========================================
# 2. SETUP LOGGING & DIRECTORIES
# ==========================================
# Create the directory FIRST, otherwise logging fails
os.makedirs(BASE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),    # Saves to modernbert/final_training_log.txt
        logging.StreamHandler(sys.stdout) # Prints to console
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
    logger.info(f"--- Starting FINAL Training Job (Logs saving to {LOG_FILE}) ---")

    # 1. Load Hugging Face Data
    try:
        ds = load_dataset(HF_DATASET_ID, HF_SUBSET)
        logger.info(f"Dataset Loaded. Splits: {ds.keys()}")
    except Exception as e:
        logger.error(f"Failed to load dataset. Did you set HF_TOKEN? Error: {e}")
        return

    # 2. Prepare Training Data (Balanced)
    df_train = ds['train'].to_pandas()
    if 'text' in df_train.columns: df_train = df_train.rename(columns={'text': 'code'})
    
    # --- BALANCING STRATEGY ---
    if 'language' in df_train.columns:
        py_count = len(df_train[df_train['language'] == 'Python'])
        if py_count > 60000:
            logger.info("Balancing: Downsampling Python to 50k...")
            df_py = df_train[df_train['language'] == 'Python'].sample(n=50000, random_state=42)
            df_others = df_train[df_train['language'] != 'Python'] # Keep ALL C++ and Java
            df_train = pd.concat([df_py, df_others]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Final Training Size: {len(df_train)}")

    # 3. Prepare Validation Data
    val_split_name = 'validation' if 'validation' in ds else 'dev'
    df_val = ds[val_split_name].to_pandas()
    if 'text' in df_val.columns: df_val = df_val.rename(columns={'text': 'code'})
    logger.info(f"Final Validation Size ({val_split_name}): {len(df_val)}")

    # 4. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    # 5. Datasets
    train_dataset = CodeDataset(df_train['code'].tolist(), df_train['label'].tolist(), tokenizer, MAX_LEN)
    val_dataset = CodeDataset(df_val['code'].tolist(), df_val['label'].tolist(), tokenizer, MAX_LEN)

    # 6. Training Args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        bf16=True,
        dataloader_num_workers=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro", 
        logging_steps=50,
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

    # 7. Train & Save
    logger.info("Starting Training...")
    trainer.train()
    
    logger.info(f"Saving Best Model to {FINAL_MODEL_DIR}...")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    # ==========================================
    # 8. TEST SET EVALUATION
    # ==========================================
    logger.info("--- Starting Evaluation on TEST Set ---")
    
    if 'test' in ds:
        df_test = ds['test'].to_pandas()
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
    else:
        logger.warning("No 'test' split found in dataset!")

    logger.info("Job Complete.")

if __name__ == "__main__":
    main()