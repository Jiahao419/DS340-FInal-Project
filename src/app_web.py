# src/app_web.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from config import PROCESSED_DIR
from predict_news import basic_clean, build_text_only_model  # reuse your existing code


app = Flask(__name__)

# Global model / retriever
clf = None
real_df = None
evi_vectorizer = None
evi_nn = None


def load_data_and_train():
    """
    Called once at startup:
    - Read all/train.csv + all/val.csv
    - Train TF-IDF + LR classifier
    - Build an evidence retriever using REAL samples from all/train
    """
    global clf, real_df, evi_vectorizer, evi_nn

    # ---------- 1. Train classifier ----------
    all_dir = PROCESSED_DIR / "all"

    train_path = all_dir / "train.csv"
    val_path = all_dir / "val.csv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Cannot find all/train.csv or all/val.csv. Please run merge_all_datasets.py and the prepare scripts first.\n"
            f"Expected paths: {train_path} / {val_path}"
        )

    print("[INIT] Loading training data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    full = pd.concat([train_df, val_df], ignore_index=True)

    if "clean_text" not in full.columns or "label" not in full.columns:
        raise ValueError("all/train.csv & all/val.csv must contain clean_text and label columns.")

    X = full["clean_text"].astype(str)
    y = full["label"].astype(int).values

    print("[INIT] Training TF-IDF + LR model on 'all' (train+val)...")
    clf = build_text_only_model()
    clf.fit(X, y)
    print("[INIT] Classifier training done ✅")

    # ---------- 2. Build evidence retriever on REAL corpus ----------
    # Use REAL samples in all/train.csv as the "real news" corpus
    print("[INIT] Building REAL evidence corpus from all/train.csv ...")
    train_df = pd.read_csv(train_path)

    if "label" not in train_df.columns or "clean_text" not in train_df.columns:
        raise ValueError("all/train.csv must contain label and clean_text columns.")

    real_df_local = train_df[train_df["label"] == 0].copy()  # 0 = REAL
    real_df_local = real_df_local.reset_index(drop=True)

    if real_df_local.empty:
        raise ValueError("No REAL samples with label==0 in all/train.csv. Cannot build evidence corpus.")

    texts = real_df_local["clean_text"].astype(str).tolist()

    # Keep TF-IDF settings consistent with build_evidence_pairs_all.py
    evi_vectorizer_local = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        norm="l2",
    )

    print("[INIT] Fitting TF-IDF on REAL corpus...")
    X_real = evi_vectorizer_local.fit_transform(texts)

    print("[INIT] Fitting NearestNeighbors (cosine) on REAL corpus...")
    evi_nn_local = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=-1)
    evi_nn_local.fit(X_real)

    # Save to globals
    real_df = real_df_local
    evi_vectorizer = evi_vectorizer_local
    evi_nn = evi_nn_local

    print(f"[INIT] REAL corpus size: {len(real_df)}")
    print("[INIT] Evidence retriever ready ✅")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    global clf, real_df, evi_vectorizer, evi_nn

    if clf is None or real_df is None:
        return jsonify({"error": "Model is not initialized yet"}), 500

    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing text field or it is empty"}), 400

    # ---------- 1. Classification ----------
    clean = basic_clean(text)
    proba_all = clf.predict_proba([clean])[0]  # [P(REAL), P(FAKE)] consistent with your dataset
    y_pred = int(proba_all.argmax())
    label_text = "FAKE" if y_pred == 1 else "REAL"

    prob_real = float(proba_all[0])
    prob_fake = float(proba_all[1])

    # ---------- 2. Retrieve one REAL evidence ----------
    # Use the same TF-IDF + NN to find the most similar REAL sample in the corpus
    q_vec = evi_vectorizer.transform([clean])
    distances, indices = evi_nn.kneighbors(q_vec, n_neighbors=1)
    dist = float(distances[0][0])
    idx = int(indices[0][0])

    sim = 1.0 - dist  # cosine distance -> cosine similarity

    evi_row = real_df.iloc[idx]
    evi_text = str(evi_row.get("clean_text", ""))
    evi_title = str(evi_row.get("title", "")) if "title" in real_df.columns else ""
    evi_source = str(evi_row.get("source", evi_row.get("source_dataset", "")))

    # Create a short snippet to avoid returning very long text
    snippet_len = 400
    evi_snippet = evi_text[:snippet_len] + ("..." if len(evi_text) > snippet_len else "")

    return jsonify(
        {
            "label": label_text,
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "evidence": {
                "title": evi_title,
                "snippet": evi_snippet,
                "similarity": sim,  # 0~1
                "source": evi_source,
            },
        }
    )


if __name__ == "__main__":
    # Run from project root: python src/app_web.py
    print("[INIT] Starting app, loading model and evidence corpus ...")
    load_data_and_train()
    # debug=True is convenient for development; turn it off in production
    app.run(host="127.0.0.1", port=5000, debug=True)
