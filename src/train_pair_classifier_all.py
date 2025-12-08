# -*- coding: utf-8 -*-
"""
train_pair_classifier_all.py

Train a binary "news + evidence" classifier using evidence pairs:

- Input: output/evidence/all/all_pairs_train.csv & all_pairs_val.csv
- Use sentence-BERT to encode (news_text, evidence_text) into vectors
- Build combined features [u, v, |u-v|, u*v, sim_score]
- Train a Logistic Regression classifier and evaluate on the val set

Run:
    python src/train_pair_classifier_all.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from sentence_transformers import SentenceTransformer

from config import OUT_DIR

EVIDENCE_DIR = OUT_DIR / "evidence" / "all"


def load_pairs(split: str) -> pd.DataFrame:
    """Load all_pairs_<split>.csv from the evidence directory."""
    path = EVIDENCE_DIR / f"all_pairs_{split}.csv"
    print(f"[load_pairs] Reading {split} pairs from {path}")
    df = pd.read_csv(path)
    if "clean_text" not in df.columns or "evidence_text" not in df.columns:
        raise ValueError(f"{path} must contain 'clean_text' and 'evidence_text'.")
    if "label" not in df.columns:
        raise ValueError(f"{path} must contain 'label' column (0=REAL,1=FAKE).")
    return df


def build_features(
    emb_news: np.ndarray,
    emb_evi: np.ndarray,
    sim_scores: np.ndarray,
) -> np.ndarray:
    """
    Build combined features:
        [u, v, |u-v|, u*v, sim_score]
    """
    diff = np.abs(emb_news - emb_evi)
    prod = emb_news * emb_evi
    sim_scores = sim_scores.reshape(-1, 1)
    feats = np.concatenate([emb_news, emb_evi, diff, prod, sim_scores], axis=1)
    return feats


def main():
    # 1) Load train / val pair data
    train_df = load_pairs("train")
    val_df = load_pairs("val")

    # X & y
    X_news_train = train_df["clean_text"].astype(str).tolist()
    X_evi_train = train_df["evidence_text"].astype(str).tolist()
    y_train = train_df["label"].astype(int).values
    sim_train = train_df["evidence_sim"].values

    X_news_val = val_df["clean_text"].astype(str).tolist()
    X_evi_val = val_df["evidence_text"].astype(str).tolist()
    y_val = val_df["label"].astype(int).values
    sim_val = val_df["evidence_sim"].values

    # 2) sentence-BERT encoding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sBERT] Using device: {device}")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[sBERT] Loading model: {model_name}")
    sbert = SentenceTransformer(model_name, device=device)

    print("[sBERT] Encoding train news text...")
    emb_news_train = sbert.encode(
        X_news_train,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("[sBERT] Encoding train evidence text...")
    emb_evi_train = sbert.encode(
        X_evi_train,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("[sBERT] Encoding val news text...")
    emb_news_val = sbert.encode(
        X_news_val,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("[sBERT] Encoding val evidence text...")
    emb_evi_val = sbert.encode(
        X_evi_val,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("[features] Building feature matrices...")
    X_train = build_features(emb_news_train, emb_evi_train, sim_train)
    X_val = build_features(emb_news_val, emb_evi_val, sim_val)

    print(f"[features] X_train shape: {X_train.shape}")
    print(f"[features] X_val   shape: {X_val.shape}")

    # 3) Train a simple Logistic Regression classifier
    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs",
    )

    print("\n[clf] Fitting pair classifier (news + evidence)...")
    clf.fit(X_train, y_train)

    print("[clf] Predicting on val set...")
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")

    print("\n=== Pair classifier (news + evidence) on 'all' / val ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=["REAL (0)", "FAKE (1)"]))


if __name__ == "__main__":
    main()
