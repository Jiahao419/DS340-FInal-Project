# -*- coding: utf-8 -*-
"""
build_evidence_pairs_all.py

Goal:
- Use all/train's REAL samples as the "real news corpus"
- Use TF-IDF + Nearest Neighbors retrieval to find, for each sample in
  all/train, all/val, all/test, its most similar REAL article
- Output a paired dataset: each row is (news_text, evidence_text, label, evidence_sim)

Output:
- output/evidence/all/all_pairs_train.csv
- output/evidence/all/all_pairs_val.csv
- output/evidence/all/all_pairs_test.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from config import PROCESSED_DIR, OUT_DIR

# Directory to store pair data
EVIDENCE_DIR = OUT_DIR / "evidence" / "all"


def load_all_split(split: str) -> pd.DataFrame:
    """Read the original all/<split>.csv data."""
    path = PROCESSED_DIR / "all" / f"{split}.csv"
    print(f"[load_all_split] Reading {split} from {path}")
    df = pd.read_csv(path, low_memory=False)
    if "clean_text" not in df.columns:
        raise ValueError(f"{path} has no 'clean_text' column.")
    if "label" not in df.columns:
        raise ValueError(f"{path} has no 'label' column.")
    return df


def build_real_corpus(train_df: pd.DataFrame):
    """
    Build the retrieval corpus from all/train by extracting all REAL samples.
    label: 0=REAL, 1=FAKE
    """
    real_df = train_df[train_df["label"] == 0].copy()
    real_df = real_df.reset_index(drop=True)

    print(f"[build_real_corpus] REAL corpus size: {len(real_df)}")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        norm="l2",
    )

    print("[build_real_corpus] Fitting TF-IDF on REAL corpus...")
    X_real = vectorizer.fit_transform(real_df["clean_text"].astype(str))

    print("[build_real_corpus] Fitting NearestNeighbors (cosine)...")
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=-1)
    nn.fit(X_real)

    return real_df, vectorizer, nn


def retrieve_evidence_for_split(
    split_name: str,
    df_split: pd.DataFrame,
    real_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    nn: NearestNeighbors,
) -> pd.DataFrame:
    """
    Retrieve evidence for all samples in a given split (train/val/test):
    - Input: df_split
    - Output: a new DataFrame with evidence_text and evidence_sim columns
    """
    print(f"\n[retrieve] Processing split = {split_name}, size={len(df_split)}")

    texts = df_split["clean_text"].astype(str).tolist()
    X_q = vectorizer.transform(texts)

    print(f"[retrieve] Running NearestNeighbors for {split_name}...")
    distances, indices = nn.kneighbors(X_q, n_neighbors=1)
    distances = distances[:, 0]  # cosine distance
    indices = indices[:, 0]

    cosine_sim = 1.0 - distances
    evidence_text = real_df.loc[indices, "clean_text"].values

    # Keep some original columns if they exist
    keep_cols = []
    for col in ["id", "title", "clean_text", "label", "label_text", "source_dataset"]:
        if col in df_split.columns:
            keep_cols.append(col)

    out_df = df_split[keep_cols].copy()
    out_df["evidence_text"] = evidence_text
    out_df["evidence_sim"] = cosine_sim
    out_df["split"] = split_name

    return out_df


def main():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load all/train, all/val, all/test
    train_df = load_all_split("train")
    val_df = load_all_split("val")
    test_df = load_all_split("test")

    # 2) Build REAL corpus (from train)
    real_df, vectorizer, nn = build_real_corpus(train_df)

    # 3) For each sample in train/val/test, retrieve one REAL evidence
    pairs_train = retrieve_evidence_for_split("train", train_df, real_df, vectorizer, nn)
    pairs_val = retrieve_evidence_for_split("val", val_df, real_df, vectorizer, nn)
    pairs_test = retrieve_evidence_for_split("test", test_df, real_df, vectorizer, nn)

    # 4) Save
    out_train = EVIDENCE_DIR / "all_pairs_train.csv"
    out_val = EVIDENCE_DIR / "all_pairs_val.csv"
    out_test = EVIDENCE_DIR / "all_pairs_test.csv"

    pairs_train.to_csv(out_train, index=False)
    pairs_val.to_csv(out_val, index=False)
    pairs_test.to_csv(out_test, index=False)

    print(f"\n[done] Saved train pairs to {out_train}")
    print(f"[done] Saved val   pairs to {out_val}")
    print(f"[done] Saved test  pairs to {out_test}")

if __name__ == "__main__":
    main()
