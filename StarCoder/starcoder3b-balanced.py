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
TRAIN_FILE = "/scratch/gilbreth/avutd01/NLP_BALANCED_DATA/balanced_train.csv"
VAL_FILE   = "/scratch/gilbreth/avutd01/NLP_BALANCED_DATA/downsized_validation.csv"

BASE_DIR        = "/scratch/gilbreth/avutd01/NLP_BALANCED_DATA"
OUTPUT_DIR      = BASE_DIR
FINAL_MODEL_DIR = BASE_DIR
LOG_FILE        = os.path.join(BASE_DIR, "final_training_log.txt")

MODEL_NAME = "bigcode/starcoder2-3b"
MAX_LEN = 4096        # safer than 8192 (which may exceed tokenizer limits)
BATCH_SIZE = 4        # safer + gradient_accumulation increases effective batch


# ==========================================
# 2. SETUP LOGGING & DIRECTORIES
# ==========================================
os.makedirs(BASE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ],
)
logger = logging.getLogger(__name__)


# ==========================================
# 3. CLEANING FUNCTION (critical fix)
# ==========================================
def clean_dataframe(df):
    # fix column rename
    if "text" in df.columns:
        df = df.rename(columns={"text": "code"})

    # remove null/empty code
    df = df[df["code"].notnull()]
    df = df[df["code"].astype(str).str.strip() != ""]

    # fix labels
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].notnull()]
    df["label"] = df["label"].astype(int)

    return df.reset_index(drop=True)


# ==========================================
# 4. DATASET CLASS (fixed version)
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

        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"  # enforce tensor output
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


# ==========================================
# 5. METRICS
# ==========================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted")
    f1_m = f1_score(labels, preds, average="macro")

    return {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m}


# ==========================================
# 6. MAIN TRAINING FUNCTION
# ==========================================
def main():
    logger.info(f"--- Starting FINAL Training Job (Logs saving to {LOG_FILE}) ---")

    # --- Load and Clean Training CSV ---
    try:
        df_train = pd.read_csv(TRAIN_FILE, engine="python", on_bad_lines="skip")
        df_train = clean_dataframe(df_train)
        logger.info(f"Training CSV Loaded. Shape after clean: {df_train.shape}")
    except Exception as e:
        logger.error(f"Failed to load training CSV: {e}")
        return

    # --- Load and Clean Validation CSV ---
    try:
        df_val = pd.read_csv(VAL_FILE, engine="python", on_bad_lines="skip")
        df_val = clean_dataframe(df_val)
        logger.info(f"Validation CSV Loaded. Shape after clean: {df_val.shape}")
    except Exception as e:
        logger.error(f"Failed to load validation CSV: {e}")
        return

    # --- Load Test Set ---
    try:
        logger.info("Loading test set from DaniilOr/SemEval-2026-Task13...")
        hf_dataset = load_dataset("DaniilOr/SemEval-2026-Task13", "A")
        df_test = hf_dataset["test"].to_pandas()
        df_test = clean_dataframe(df_test)
        logger.info("HuggingFace test dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load test dataset: {e}")
        return

    logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        pad_token_id=tokenizer.pad_token_id
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- Dataset Objects ---
    train_dataset = CodeDataset(df_train["code"].tolist(), df_train["label"].tolist(), tokenizer, MAX_LEN)
    val_dataset   = CodeDataset(df_val["code"].tolist(), df_val["label"].tolist(), tokenizer, MAX_LEN)

    # --- Data Collator ---
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        dataloader_num_workers=4,
        logging_steps=50,
        report_to="none",
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- Train ---
    logger.info("Starting Training...")
    trainer.train()

    # --- Save Model ---
    logger.info(f"Saving Best Model to {FINAL_MODEL_DIR}...")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    pt_file = os.path.join(FINAL_MODEL_DIR, "starcoder2_3b_finetuned.pt")
    torch.save(model.state_dict(), pt_file)
    logger.info(f"Model saved as .pt file: {pt_file}")

    # --- Test Evaluation ---
    logger.info("--- Starting Evaluation on TEST Set ---")
    test_dataset = CodeDataset(
        df_test["code"].tolist(),
        df_test["label"].tolist(),
        tokenizer,
        MAX_LEN
    )

    test_results = trainer.predict(test_dataset)
    logger.info("===== FINAL TEST RESULTS =====")
    logger.info(f"Accuracy:  {test_results.metrics['test_accuracy']:.4f}")
    logger.info(f"Macro F1:  {test_results.metrics['test_f1_macro']:.4f}")
    logger.info("================================")

    # Save predictions
    predictions = test_results.predictions.argmax(-1)
    df_test["predicted_label"] = predictions
    prediction_file = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    df_test.to_csv(prediction_file, index=False)
    logger.info(f"Predictions saved to {prediction_file}")


if __name__ == "__main__":
    main()
