import pandas as pd
import numpy as np
import re
import math
import os
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import load_dataset

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_FILE = "balanced_train.csv"
VAL_FILE   = "downsized_validation.csv"
OUTPUT_FILE = "submission_stylometry_advanced.csv"

# ==========================================
# 1. ADVANCED FEATURE EXTRACTOR
# ==========================================
def calculate_entropy(text):
    if not text: return 0
    counts = Counter(text)
    length = len(text)
    return -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())

def extract_features(text):
    code = str(text) if pd.notna(text) else ""
    lines = code.split('\n')
    words = code.split()
    chars = list(code)

    # --- A. Entropy (The "Chaos" Factor) ---
    char_entropy = calculate_entropy(code)

    # --- B. Structure & Layout ---
    line_count = len(lines)
    avg_line_len = np.mean([len(l) for l in lines]) if lines else 0
    empty_lines = sum(1 for l in lines if not l.strip())
    empty_ratio = empty_lines / line_count if line_count > 0 else 0

    # Indentation Variance (Machine code is usually very uniform)
    indent_lengths = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    indent_std = np.std(indent_lengths) if indent_lengths else 0

    # --- C. Vocabulary ---
    # Unique Word Ratio (Humans use more varied names)
    unique_ratio = len(set(words)) / len(words) if words else 0

    # Symbol Density (Machines might over-use specific syntax)
    brackets = (code.count('{') + code.count('(') + code.count('[')) / len(chars) if chars else 0
    semicolons = code.count(';') / len(chars) if chars else 0

    return [char_entropy, avg_line_len, indent_std, empty_ratio, unique_ratio, brackets, semicolons]

FEATURE_NAMES = ["Entropy", "Avg Line Len", "Indent Std", "Empty Ratio", "Unique Ratio", "Bracket Density", "Semi Density"]

# ==========================================
# 2. MAIN EXECUTION
# ==========================================
def main():
    print("--- 🧠 Starting Advanced Stylometry ---")

    # 1. Load Data
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return

    df_train = pd.read_csv(TRAIN_FILE)
    df_val = pd.read_csv(VAL_FILE)

    print(f"Training on {len(df_train)} samples...")
    X_train = np.array(df_train['code'].apply(extract_features).tolist())
    y_train = df_train['label'].tolist()

    X_val = np.array(df_val['code'].apply(extract_features).tolist())
    y_val = df_val['label'].tolist()

    # 2. Train Gradient Boosting (Better than Random Forest for these features)
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # 3. Evaluate
    val_preds = clf.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds, average='macro')

    print("\n" + "="*30)
    print("   VALIDATION RESULTS   ")
    print("="*30)
    print(f"Accuracy: {acc:.2%}")
    print(f"Macro F1: {f1:.4f}")

    # Feature Importance
    print("\n--- Feature Importance ---")
    for name, imp in zip(FEATURE_NAMES, clf.feature_importances_):
        print(f"{name:<15}: {imp:.4f}")

    # 4. Predict on Test Set
    print("\n--- predicting on Test Set ---")
    hf_ds = load_dataset("DaniilOr/SemEval-2026-Task13", "A", split="test")
    df_test = hf_ds.to_pandas()
    if 'text' in df_test.columns: df_test = df_test.rename(columns={'text': 'code'})

    X_test = np.array(df_test['code'].apply(extract_features).tolist())
    test_preds = clf.predict(X_test)

    # Check Accuracy (if labels exist)
    if 'label' in df_test.columns:
        y_test_true = df_test['label'].tolist()
        test_acc = accuracy_score(y_test_true, test_preds)
        print(f"TEST ACCURACY: {test_acc:.2%}")

if __name__ == "__main__":
    main()