import argparse
from collections import Counter

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support

from config import FEATURES_DIR, DATASETS


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

    in_dir = FEATURES_DIR / args.dataset

    train_path = in_dir / "train_fe.csv"
    val_path = in_dir / "val_fe.csv"

    print("Train path:", train_path)
    print("Val path:", val_path)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    print("Train shape:", train.shape)
    print("Val shape:", val.shape)

    if "clean_text" not in train.columns or "label" not in train.columns:
        raise ValueError("train_fe/val_fe need conclude clean_text and label")

    X_train = train.copy()
    y_train = train["label"].values

    X_val = val.copy()
    y_val = val["label"].values

    if "label_text" in train.columns:
        print("Label distribution (train):")
        print(train["label_text"].value_counts(normalize=True))

    # Majority baseline
    counter = Counter(y_train)
    majority_label, majority_count = counter.most_common(1)[0]

    print(f"\n=== Majority baseline ({args.dataset}, with features) ===")
    print("Majority label (0=REAL,1=FAKE):", majority_label)
    print("Majority proportion in train:", majority_count / len(y_train))

    y_val_maj = [majority_label] * len(y_val)
    maj_acc = accuracy_score(y_val, y_val_maj)
    maj_f1 = f1_score(y_val, y_val_maj, average="macro")

    print("Val accuracy:", maj_acc)
    print("Val macro F1:", maj_f1)

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
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_val.columns:
            X_val[col] = 0

    X_train[numeric_features] = X_train[numeric_features].fillna(0)
    X_val[numeric_features] = X_val[numeric_features].fillna(0)

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.9,
                sublinear_tf=True,
            ), "clean_text"),
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )

    # Set class_weight for different datasets
    if args.dataset == "all":
        class_weight = "balanced"
    else:
        class_weight = None

    print(f"\n=== TF-IDF(text) + numeric features + Logistic Regression ({args.dataset}) ===")

    model = Pipeline([
        ("preprocess", preprocess),
        ("logreg", LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight=class_weight,
        )),
    ])

    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

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

    # Threshold sweep on validation set (using predicted probabilities)
    if hasattr(model.named_steps["logreg"], "predict_proba"):
        y_val_proba = model.predict_proba(X_val)[:, 1]  # probability of class 1 (FAKE)

        print("\n=== Threshold sweep on validation set ===")
        print("thr\tacc\tmacro_f1\treal_rec\tfake_rec")

        for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            y_thr = (y_val_proba >= thr).astype(int)

            acc_thr = accuracy_score(y_val, y_thr)
            macro_f1_thr = f1_score(y_val, y_thr, average="macro")

            _, recall, _, _ = precision_recall_fscore_support(
                y_val, y_thr, labels=[0, 1], zero_division=0
            )
            real_rec, fake_rec = recall[0], recall[1]

            print(f"{thr:.2f}\t{acc_thr:.4f}\t{macro_f1_thr:.4f}\t{real_rec:.4f}\t{fake_rec:.4f}")


if __name__ == "__main__":
    main()
