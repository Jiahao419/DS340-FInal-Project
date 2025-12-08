# -*- coding: utf-8 -*-
"""
bert_news_classifier.py

Fine-tune or evaluate a BERT-like model (default: distilbert-base-uncased)
on our fake-news datasets (main / covid / welfake / all).

Usage:
  Train + evaluate:
    python src/bert_news_classifier.py --dataset all

  Evaluate only with an existing checkpoint + threshold scan (no further training):
    python src/bert_news_classifier.py --dataset all --skip_train

Optional: limit number of training samples, adjust epochs / batch_size:
    python src/bert_news_classifier.py --dataset all --max_train_samples 20000 --epochs 2 --batch_size 8
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    precision_recall_fscore_support,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from config import PROCESSED_DIR, OUT_DIR, DATASETS


# --------- PyTorch Dataset wrapper ---------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        texts: list of strings
        labels: list/array of ints (0 or 1)
        """
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# --------- Metrics for Trainer ---------
def compute_metrics(eval_pred):
    """
    eval_pred: transformers.EvalPrediction
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=DATASETS,
        help="Which dataset split folder under output/processed/ to use.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max sequence length for BERT tokenizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional: cap number of training samples for quicker experiments",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="If set, skip training and only evaluate using existing checkpoint.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---------- Load data ----------
    ds_dir = PROCESSED_DIR / args.dataset
    train_path = ds_dir / "train.csv"
    val_path = ds_dir / "val.csv"

    print("Train path:", train_path)
    print("Val path:", val_path)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)

    if "clean_text" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("train/val csv must contain 'clean_text' and 'label' columns.")

    # Optional subsampling for faster experiments
    if (
        args.max_train_samples is not None
        and args.max_train_samples < len(train_df)
    ):
        train_df = train_df.sample(n=args.max_train_samples, random_state=42)
        train_df = train_df.reset_index(drop=True)
        print(f"Subsampled train to {len(train_df)} samples.")

    X_train = train_df["clean_text"].astype(str).tolist()
    y_train = train_df["label"].astype(int).tolist()

    X_val = val_df["clean_text"].astype(str).tolist()
    y_val = val_df["label"].astype(int).tolist()

    if "label_text" in train_df.columns:
        print("Label distribution (train):")
        print(train_df["label_text"].value_counts(normalize=True))

    # ---------- Tokenizer & Dataset ----------
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = NewsDataset(
        texts=X_train,
        labels=y_train,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_dataset = NewsDataset(
        texts=X_val,
        labels=y_val,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # ---------- Model / checkpoint loading ----------
    output_dir = OUT_DIR / f"bert_{args.dataset}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)

    if args.skip_train:
        ckpt_root = Path(output_dir)
        ckpts = sorted(
            ckpt_root.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if ckpts:
            load_path = ckpts[-1]
            print(f"\n[skip_train] Loading model from checkpoint: {load_path}")
        elif (ckpt_root / "pytorch_model.bin").exists():
            load_path = ckpt_root
            print(f"\n[skip_train] Loading model from output_dir: {load_path}")
        else:
            load_path = args.model_name
            print(
                "\n[skip_train] WARNING: No checkpoint found, loading base model. "
                "Results will not match previous training!"
            )

        model = AutoModelForSequenceClassification.from_pretrained(
            str(load_path),
            num_labels=2,
        )
        model.to(device)
    else:
        print(f"\nLoading model: {args.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
        )
        model.to(device)

    # ---------- TrainingArguments ----------
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=not args.skip_train,
        do_eval=True,
        eval_strategy="epoch",  
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None if args.skip_train else train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ---------- Train  ----------
    if not args.skip_train:
        print("\n***** Start training *****")
        trainer.train()
        trainer.save_model(output_dir)

    # ---------- Evaluate ----------
    print("\n***** Evaluate on validation *****")
    eval_results = trainer.evaluate()
    print("Eval results:", eval_results)

    # Extra: detailed classification report
    print("\n***** Detailed classification report on validation *****")
    preds_output = trainer.predict(val_dataset)
    logits = preds_output.predictions
    preds = np.argmax(logits, axis=-1)

    print("Validation accuracy:",
          accuracy_score(y_val, preds))
    print("Validation macro F1:",
          f1_score(y_val, preds, average="macro"))

    print("\nClassification report:")
    print(classification_report(
        y_val,
        preds,
        target_names=["REAL (0)", "FAKE (1)"],
    ))

    # ---------- Threshold sweep ----------
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    fake_prob = probs[:, 1]  # P(FAKE)

    print("\n=== Threshold sweep on validation set ===")
    print("thr\tacc\tmacro_f1\treal_rec\tfake_rec")

    for thr in np.arange(0.30, 0.71, 0.05):
        preds_thr = (fake_prob >= thr).astype(int)  # >= thr 判 FAKE
        acc = accuracy_score(y_val, preds_thr)
        macro_f1 = f1_score(y_val, preds_thr, average="macro")

        _, recall, _, _ = precision_recall_fscore_support(
            y_val,
            preds_thr,
            labels=[0, 1],
            zero_division=0,
        )
        real_rec, fake_rec = recall
        print(
            f"{thr:.2f}\t{acc:.4f}\t{macro_f1:.4f}\t"
            f"{real_rec:.4f}\t{fake_rec:.4f}"
        )


if __name__ == "__main__":
    main()
