# -*- coding: utf-8 -*-
"""
analyze_all_diagnostics.py

Deep-dive diagnostics on the merged "all" dataset:

- Confusion matrices for LR+TFIDF+features and BERT
- ROC curves (FAKE vs REAL) for LR and BERT
- Precision–Recall curves (FAKE as positive)
- Top tokens for LR model (words pushing towards FAKE / REAL)

Run from project root (with .venv_torch activated):

    python src/analyze_all_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import PROCESSED_DIR, FEATURES_DIR, OUT_DIR


# ----------------------- Helpers -----------------------


def load_all_feature_data():
    """Load train/val features for 'all' dataset."""
    in_dir = FEATURES_DIR / "all"
    train_path = in_dir / "train_fe.csv"
    val_path = in_dir / "val_fe.csv"

    print("LR features train path:", train_path)
    print("LR features val path  :", val_path)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    print("Train shape:", train.shape)
    print("Val shape  :", val.shape)

    # labels: 0 = REAL, 1 = FAKE
    y_train = train["label"].astype(int).values
    y_val = val["label"].astype(int).values

    return train, val, y_train, y_val


def train_lr_tfidf_feats(train_df, val_df, y_train):
    """
    Train LR + TF-IDF + numeric features on 'all' dataset.

    Returns:
        model (Pipeline), y_val_pred, y_val_proba
    """
    # numeric feature names must match add_features.py / baseline_lr_plus_feats.py
    numeric_features = [
        "num_chars",
        "num_words",
        "avg_word_len",
        "num_exclamation",
        "num_question",
        "unique_words",
        "lexical_richness",
        "num_stopwords",
        "stopword_ratio",
        "num_digits",
        "digit_ratio",
        "num_urls",
        "year",
        "month",
    ]

    for col in numeric_features:
        if col not in train_df:
            train_df[col] = 0
            val_df[col] = 0

    train_df[numeric_features] = train_df[numeric_features].fillna(0)
    val_df[numeric_features] = val_df[numeric_features].fillna(0)

    # Text vectorizer (same style as baseline)
    text_vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
    )

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler(with_mean=False))]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("text", text_vectorizer, "clean_text"),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", clf),
        ]
    )

    print("\n[LR] Fitting LR+TFIDF+features on 'all' train set...")
    model.fit(train_df, y_train)

    print("[LR] Predicting on validation set...")
    y_val_pred = model.predict(val_df)
    y_val_proba = model.predict_proba(val_df)[:, 1]  # P(FAKE)

    return model, y_val_pred, y_val_proba


from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ...

def get_bert_predictions_all():
    """
    Load BERT checkpoint for 'all' and run inference on val set.

    Returns:
        y_val (np.array), y_pred (np.array), y_proba_fake (np.array)
    """
    val_path = PROCESSED_DIR / "all" / "val.csv"
    print("\n[BERT] Val path:", val_path)

    val_df = pd.read_csv(val_path)
    print("[BERT] Val shape:", val_df.shape)

    texts = val_df["clean_text"].astype(str).tolist()
    y_val = val_df["label"].astype(int).values  # 0 = REAL, 1 = FAKE

    bert_root = OUT_DIR / "bert_all"

    checkpoint_dirs = [
        d for d in bert_root.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]

    if not checkpoint_dirs:
        raise RuntimeError(f"No checkpoint-* folders found under {bert_root}")

    checkpoint_dirs.sort(key=lambda p: int(p.name.split("-")[-1]))
    bert_dir = checkpoint_dirs[-1]

    print("[BERT] Loading checkpoint from:", bert_dir)

    MODEL_NAME = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(bert_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[BERT] Using device:", device)
    model.to(device)
    model.eval()

    all_probs = []
    all_preds = []

    batch_size = 32
    n = len(texts)
    for start in range(0, n, batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)  # (batch, 2)

        probs_np = probs.detach().cpu().numpy()
        all_probs.append(probs_np)
        all_preds.append(probs_np.argmax(axis=1))

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # probability of FAKE (label 1)
    y_proba_fake = all_probs[:, 1]

    return y_val, all_preds, y_proba_fake


def plot_confusion_matrices(y_true_lr, y_pred_lr, y_true_bert, y_pred_bert):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    cm_lr = confusion_matrix(y_true_lr, y_pred_lr)
    disp_lr = ConfusionMatrixDisplay(
        confusion_matrix=cm_lr,
        display_labels=["REAL (0)", "FAKE (1)"],
    )
    disp_lr.plot(ax=axes[0], values_format="d", colorbar=False)
    axes[0].set_title("LR+TFIDF+features (all)")

    cm_bert = confusion_matrix(y_true_bert, y_pred_bert)
    disp_bert = ConfusionMatrixDisplay(
        confusion_matrix=cm_bert,
        display_labels=["REAL (0)", "FAKE (1)"],
    )
    disp_bert.plot(ax=axes[1], values_format="d", colorbar=False)
    axes[1].set_title("BERT (all)")

    fig.suptitle("Confusion matrices on 'all' (val)")
    fig.tight_layout()
    plt.show()


def plot_roc_and_pr(
    y_true_lr,
    proba_lr,
    y_true_bert,
    proba_bert,
):
    # ROC
    fpr_lr, tpr_lr, _ = roc_curve(y_true_lr, proba_lr)
    fpr_b, tpr_b, _ = roc_curve(y_true_bert, proba_bert)

    auc_lr = auc(fpr_lr, tpr_lr)
    auc_b = auc(fpr_b, tpr_b)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_lr, tpr_lr, label=f"LR+feats (AUC={auc_lr:.3f})")
    plt.plot(fpr_b, tpr_b, label=f"BERT (AUC={auc_b:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate (FAKE)")
    plt.ylabel("True Positive Rate (FAKE)")
    plt.title("ROC curve on 'all' (FAKE as positive)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Precision–Recall (FAKE positive)
    prec_lr, rec_lr, _ = precision_recall_curve(y_true_lr, proba_lr)
    prec_b, rec_b, _ = precision_recall_curve(y_true_bert, proba_bert)

    pr_auc_lr = auc(rec_lr, prec_lr)
    pr_auc_b = auc(rec_b, prec_b)

    plt.figure(figsize=(6, 5))
    plt.plot(rec_lr, prec_lr, label=f"LR+feats (AUPRC={pr_auc_lr:.3f})")
    plt.plot(rec_b, prec_b, label=f"BERT (AUPRC={pr_auc_b:.3f})")
    plt.xlabel("Recall (FAKE)")
    plt.ylabel("Precision (FAKE)")
    plt.title("Precision–Recall curve on 'all' (FAKE positive)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_lr_top_tokens(lr_model, top_k: int = 15):
    """
    Extract top positive/negative tokens from LR classifier
    (tokens with largest positive / negative coefficients).
    """
    # Get vectorizer & classifier from pipeline
    preprocess = lr_model.named_steps["preprocess"]
    clf = lr_model.named_steps["clf"]

    text_vect = preprocess.named_transformers_["text"]
    feature_names = np.array(text_vect.get_feature_names_out())

    # LR coef shape: (1, n_features_total). We only care about text part,
    # which comes first in the sparse matrix from ColumnTransformer.
    coef = clf.coef_[0]
    n_text_feats = len(feature_names)
    text_coef = coef[:n_text_feats]

    # Top tokens pushing towards FAKE (label 1) and REAL (label 0)
    top_fake_idx = np.argsort(text_coef)[-top_k:][::-1]
    top_real_idx = np.argsort(text_coef)[:top_k]

    fake_tokens = feature_names[top_fake_idx]
    fake_weights = text_coef[top_fake_idx]

    real_tokens = feature_names[top_real_idx]
    real_weights = text_coef[top_real_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    axes[0].barh(range(top_k), real_weights[::-1])
    axes[0].set_yticks(range(top_k))
    axes[0].set_yticklabels(real_tokens[::-1])
    axes[0].set_title(f"Top {top_k} tokens → REAL (negative weights)")
    axes[0].invert_xaxis()

    axes[1].barh(range(top_k), fake_weights[::-1])
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels(fake_tokens[::-1])
    axes[1].set_title(f"Top {top_k} tokens → FAKE (positive weights)")

    fig.suptitle("Important tokens for LR+TFIDF+features (all)")
    fig.tight_layout()
    plt.show()


# ----------------------- Main -----------------------


def main():
    # 1) LR: train & predict on 'all'
    train_df, val_df, y_train, y_val_lr = load_all_feature_data()
    lr_model, y_pred_lr, proba_lr = train_lr_tfidf_feats(train_df, val_df, y_train)

    # 2) BERT: load checkpoint & predict on 'all'
    y_val_bert, y_pred_bert, proba_bert = get_bert_predictions_all()

    # 3) Confusion matrices
    plot_confusion_matrices(y_val_lr, y_pred_lr, y_val_bert, y_pred_bert)

    # 4) ROC & PR curves
    plot_roc_and_pr(y_val_lr, proba_lr, y_val_bert, proba_bert)

    # 5) Top tokens from LR
    plot_lr_top_tokens(lr_model, top_k=15)


if __name__ == "__main__":
    main()
