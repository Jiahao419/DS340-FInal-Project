# -*- coding: utf-8 -*-
"""
visualize_results.py

Aggregate and visualize final results for the DS340 project:
1. Label distribution (REAL vs FAKE) for each dataset
2. Accuracy comparison of different models on each dataset
3. Macro F1 comparison of different models on each dataset
4. On the 'all' dataset, compare REAL/FAKE precision & recall between LR+features and BERT
5. On the 'all' dataset, plot BERT performance curves (macro F1 + REAL/FAKE recall) under different thresholds

Output figures are saved to the output/ directory by default.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DIR, OUT_DIR, DATASETS


RESULTS_CSV = OUT_DIR / "results_summary.csv"
BERT_THRESH_CSV = OUT_DIR / "bert_threshold_all.csv"


def plot_label_distribution():
    """
    Fig 1: Label distribution (REAL / FAKE proportions) for each dataset.
    Use label_text from train.csv for counting.
    """
    rows = []
    for ds in DATASETS:
        train_path = PROCESSED_DIR / ds / "train.csv"
        if not train_path.exists():
            print(f"[label_dist] Skip {ds}, no train.csv at {train_path}")
            continue

        df = pd.read_csv(train_path)
        if "label_text" not in df.columns:
            print(f"[label_dist] Skip {ds}, no 'label_text' column")
            continue

        counts = df["label_text"].value_counts()
        for label, cnt in counts.items():
            rows.append(
                {"dataset": ds, "label": label, "count": cnt}
            )

    if not rows:
        print("[label_dist] No data to plot.")
        return

    dist_df = pd.DataFrame(rows)
    dist_df["proportion"] = dist_df["count"] / dist_df.groupby("dataset")["count"].transform("sum")

    pivot = dist_df.pivot(index="dataset", columns="label", values="proportion").fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Proportion in train set")
    ax.set_title("Label distribution by dataset (train)")
    ax.legend(title="Label")
    plt.tight_layout()

    out_path = OUT_DIR / "fig_label_distribution.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[label_dist] Saved to {out_path}")


def plot_metric_by_model(metric: str, filename: str, title: str):
    """
    Fig 2A / 2B: accuracy / macro_f1 comparison of different models on each dataset.
    metric: "accuracy" or "macro_f1"
    """
    if not RESULTS_CSV.exists():
        print(f"[metric_by_model] {RESULTS_CSV} not found, skip.")
        return

    df = pd.read_csv(RESULTS_CSV)

    df_m = df[df["metric"] == metric].copy()
    if df_m.empty:
        print(f"[metric_by_model] No rows for metric={metric}")
        return

    pivot = df_m.pivot(index="dataset", columns="model", values="value")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(title="Model")
    plt.tight_layout()

    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[metric_by_model] Saved {metric} figure to {out_path}")


def plot_precision_recall_all():
    """
    Fig 3: On the 'all' dataset, compare REAL/FAKE precision & recall
    between LR+features and BERT.

    We only look at rows where dataset == "all" and
    metric in {real_precision, real_recall, fake_precision, fake_recall}.
    """
    if not RESULTS_CSV.exists():
        print(f"[prec_rec_all] {RESULTS_CSV} not found, skip.")
        return

    df = pd.read_csv(RESULTS_CSV)
    metrics = ["real_precision", "real_recall", "fake_precision", "fake_recall"]

    df_all = df[(df["dataset"] == "all") & (df["metric"].isin(metrics))]
    if df_all.empty:
        print("[prec_rec_all] No rows for dataset=all with precision/recall metrics.")
        return

    pivot = df_all.pivot(index="metric", columns="model", values="value")

    # Keep only the two models we care about (if they exist)
    preferred_models = ["lr_tfidf_feats", "bert"]
    cols = [m for m in preferred_models if m in pivot.columns]
    if cols:
        pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall for REAL & FAKE on all (LR+features vs BERT)")
    ax.legend(title="Model")
    plt.tight_layout()

    out_path = OUT_DIR / "fig_all_precision_recall_by_model.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[prec_rec_all] Saved to {out_path}")


def plot_bert_threshold_curve():
    """
    Fig 4: On the 'all' dataset, plot macro F1 and REAL/FAKE recall
    as functions of decision threshold for BERT.

    Read bert_threshold_all.csv with columns:
        thr, acc, macro_f1, real_rec, fake_rec
    """
    if not BERT_THRESH_CSV.exists():
        print(f"[bert_threshold] {BERT_THRESH_CSV} not found, skip.")
        return

    df = pd.read_csv(BERT_THRESH_CSV)
    required_cols = {"thr", "acc", "macro_f1", "real_rec", "fake_rec"}
    if not required_cols.issubset(df.columns):
        print(f"[bert_threshold] Missing columns in {BERT_THRESH_CSV}, got {df.columns}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["thr"], df["macro_f1"], marker="o", label="Macro F1")
    ax.plot(df["thr"], df["real_rec"], marker="o", label="REAL recall")
    ax.plot(df["thr"], df["fake_rec"], marker="o", label="FAKE recall")

    ax.set_xlabel("Decision threshold for FAKE (P(FAKE) >= thr)")
    ax.set_ylabel("Score")
    ax.set_title("BERT on all: threshold vs macro F1 & recall")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_path = OUT_DIR / "fig_bert_threshold_curve_all.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[bert_threshold] Saved to {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_label_distribution()
    plot_metric_by_model(
        metric="accuracy",
        filename="fig_accuracy_by_dataset_and_model.png",
        title="Validation accuracy by dataset and model",
    )
    plot_metric_by_model(
        metric="macro_f1",
        filename="fig_macro_f1_by_dataset_and_model.png",
        title="Validation macro F1 by dataset and model",
    )
    plot_precision_recall_all()
    plot_bert_threshold_curve()


if __name__ == "__main__":
    main()
