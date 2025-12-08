import argparse
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import PROCESSED_DIR, DATASETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="main",
        choices=DATASETS,
        help="Dataset to train on (main/covid/welfake/all)",
    )
    args = parser.parse_args()

    in_dir = PROCESSED_DIR / args.dataset

    train_path = in_dir / "train.csv"
    val_path = in_dir / "val.csv"

    print("Train path:", train_path)
    print("Val path:", val_path)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    print("Train shape:", train.shape)
    print("Val shape:", val.shape)

    if "clean_text" not in train.columns or "label" not in train.columns:
        raise ValueError("train/val must contain clean_text and label columns")

    X_train_text = train["clean_text"].astype(str)
    y_train = train["label"].values

    X_val_text = val["clean_text"].astype(str)
    y_val = val["label"].values

    if "label_text" in train.columns:
        print("Label distribution (train):")
        print(train["label_text"].value_counts(normalize=True))

    # 1. Majority baseline
    counter = Counter(y_train)
    majority_label, majority_count = counter.most_common(1)[0]

    print(f"\n=== Majority baseline ({args.dataset}) ===")
    print("Majority label (0=REAL,1=FAKE):", majority_label)
    print("Majority proportion in train:", majority_count / len(y_train))

    y_val_maj = [majority_label] * len(y_val)
    maj_acc = accuracy_score(y_val, y_val_maj)
    maj_f1 = f1_score(y_val, y_val_maj, average="macro")

    print("Val accuracy:", maj_acc)
    print("Val macro F1:", maj_f1)

    # 2. TF-IDF + Logistic Regression
    print(f"\n=== TF-IDF + Logistic Regression baseline ({args.dataset}) ===")

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.9,
            sublinear_tf=True,
        )),
        ("logreg", LogisticRegression(
            max_iter=200,
            n_jobs=-1,
            class_weight=None,
        )),
    ])

    clf.fit(X_train_text, y_train)
    y_val_pred = clf.predict(X_val_text)

    acc = accuracy_score(y_val, y_val_pred)
    macro_f1 = f1_score(y_val, y_val_pred, average="macro")

    print("Val accuracy:", acc)
    print("Val macro F1:", macro_f1)
    print("\nClassification report:")
    print(classification_report(
        y_val,
        y_val_pred,
        target_names=["REAL (0)", "FAKE (1)"],
    ))


if __name__ == "__main__":
    main()
