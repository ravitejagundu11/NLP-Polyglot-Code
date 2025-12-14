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

BASE_DIR = "/scratch/gilbreth/avutd01/NLP/starcoder-7b"
OUTPUT_DIR = BASE_DIR
FINAL_MODEL_DIR = BASE_DIR
LOG_FILE = os.path.join(BASE_DIR, "final_training_log.txt")

# ✅ UPDATED MODEL (7B)
MODEL_NAME = "bigcode/starcoder2-7b"

# ✅ Increased for faster training on A100-80GB
MAX_LEN = 8192

# ✅ Increased for faster training - A100-80GB can handle this
BATCH_SIZE = 8

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
    def __init__(self, texts, labels, tokenizer, max_len=2048):
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
        except Exception:
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
    logger.info(f"--- Starting FINAL Training Job (StarCoder2-7B) ---")

    # 1. Load Dataset
    try:
        ds = load_dataset(HF_DATASET_ID, HF_SUBSET)
        logger.info(f"Dataset Loaded. Splits: {ds.keys()}")
    except Exception as e:
        logger.error(f"Dataset Load Failed: {e}")
        return

    # 2. Prepare Training Data (Balanced)
    df_train = ds['train'].to_pandas()
    if 'text' in df_train.columns:
        df_train = df_train.rename(columns={'text': 'code'})

    if 'language' in df_train.columns:
        py_count = len(df_train[df_train['language'] == 'Python'])
        if py_count > 60000:
            logger.info("Downsampling Python to 50k...")
            df_py = df_train[df_train['language'] == 'Python'].sample(n=50000, random_state=42)
            df_others = df_train[df_train['language'] != 'Python']
            df_train = pd.concat([df_py, df_others]).sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Final Training Size: {len(df_train)}")

    # 3. Validation Data
    val_split_name = 'validation' if 'validation' in ds else 'dev'
    df_val = ds[val_split_name].to_pandas()
    if 'text' in df_val.columns:
        df_val = df_val.rename(columns={'text': 'code'})

    logger.info(f"Validation Size: {len(df_val)}")

    # 4. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        pad_token_id=tokenizer.pad_token_id,
        low_cpu_mem_usage=True
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # ✅ Enable gradient checkpointing for 7B
    model.gradient_checkpointing_enable()

    # 5. Datasets
    train_dataset = CodeDataset(df_train['code'].tolist(), df_train['label'].tolist(), tokenizer, MAX_LEN)
    val_dataset = CodeDataset(df_val['code'].tolist(), df_val['label'].tolist(), tokenizer, MAX_LEN)

    # 6. Training Args (7B SAFE)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,   # ✅ Lower LR for big models
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=16,   # ✅ Restores effective batch
        gradient_checkpointing=True,
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

    logger.info("Saving Best Model...")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    pt_file = os.path.join(FINAL_MODEL_DIR, "starcoder2_7b_finetuned.pt")
    torch.save(model.state_dict(), pt_file)
    logger.info(f"Model saved at: {pt_file}")

    # ==========================================
    # 8. TEST SET EVALUATION
    # ==========================================
    if 'test' in ds:
        df_test = ds['test'].to_pandas()
        if 'text' in df_test.columns:
            df_test = df_test.rename(columns={'text': 'code'})

        test_dataset = CodeDataset(
            df_test['code'].tolist(),
            df_test['label'].tolist(),
            tokenizer,
            MAX_LEN
        )

        test_results = trainer.predict(test_dataset)

        logger.info("✅ FINAL TEST RESULTS ✅")
        logger.info(f"Test Accuracy: {test_results.metrics['test_accuracy']:.4f}")
        logger.info(f"Test Macro F1: {test_results.metrics['test_f1_macro']:.4f}")

        predictions = test_results.predictions.argmax(-1)
        df_test['predicted_label'] = predictions

        prediction_file = os.path.join(OUTPUT_DIR, "test_predictions.csv")
        df_test.to_csv(prediction_file, index=False)
        logger.info(f"Predictions saved at: {prediction_file}")

    logger.info("✅ JOB COMPLETE ✅")

if __name__ == "__main__":
    main()
