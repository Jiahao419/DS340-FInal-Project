import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import PROCESSED_DIR, FEATURES_DIR


def basic_clean(text: str) -> str:
    """Keep consistent with earlier: lowercase + collapse extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = " ".join(text.split())
    return text


def build_text_only_model():
    """TF-IDF on clean_text only + Logistic Regression."""
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.9,
            sublinear_tf=True,
        )),
        ("logreg", LogisticRegression(
            max_iter=300,
            n_jobs=-1,
            class_weight=None,
        )),
    ])
    return clf


def build_text_plus_feats_model(numeric_features):
    """TF-IDF on clean_text + numeric features + Logistic Regression."""
    preprocess = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.9,
                sublinear_tf=True,
            ), "clean_text"),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    clf = Pipeline([
        ("preprocess", preprocess),
        ("logreg", LogisticRegression(
            max_iter=300,
            n_jobs=-1,
            class_weight=None,
        )),
    ])
    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",   # default: use merged "all"
        choices=["main", "covid", "welfake", "all"],
        help="Which dataset to train on (main/covid/welfake/all)",
    )
    parser.add_argument(
        "--use_features",
        action="store_true",
        help="Whether to use handcrafted features (num_chars, etc.). If set, read *_fe.csv from features directory.",
    )
    args = parser.parse_args()

    if args.use_features:
        in_dir = FEATURES_DIR / args.dataset
        train_path = in_dir / "train_fe.csv"
        val_path = in_dir / "val_fe.csv"
        print("Using: text + handcrafted features")
    else:
        in_dir = PROCESSED_DIR / args.dataset
        train_path = in_dir / "train.csv"
        val_path = in_dir / "val.csv"
        print("Using: text only (clean_text)")

    print("Train path:", train_path)
    print("Val path  :", val_path)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    print("Train shape:", train.shape)
    print("Val shape  :", val.shape)

    # Merge train + val to train the final model (for deployment; no separate validation set)
    full = pd.concat([train, val], ignore_index=True)

    if "clean_text" not in full.columns or "label" not in full.columns:
        raise ValueError("Dataset must contain clean_text and label columns")

    X = full.copy()
    y = full["label"].values

    if args.use_features:
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
        # Ensure all numeric feature columns exist
        for col in numeric_features:
            if col not in X.columns:
                X[col] = 0
        X[numeric_features] = X[numeric_features].fillna(0)

        model = build_text_plus_feats_model(numeric_features)
    else:
        model = build_text_only_model()

    print("\nStart training final model (train+val)...")
    model.fit(X, y)
    print("Training finished ✅")

    # =============== Interactive prediction section ===============
    print("\nEntering interactive prediction mode. You can input a news article and I will predict REAL/FAKE.")
    print("Press Enter on an empty line to exit.\n")

    while True:
        s = input("Please enter a piece of news (or press Enter to exit):\n> ").strip()
        if not s:
            print("Exiting prediction mode.")
            break

        s_clean = basic_clean(s)
        y_pred = model.predict([s_clean])[0]
        proba_all = model.predict_proba([s_clean])[0]
        proba = proba_all[y_pred]

        label_text = "FAKE" if y_pred == 1 else "REAL"
        print(f"\nPrediction: {label_text}  (confidence ≈ {proba:.3f})")
        print(f"Class probabilities: REAL={proba_all[0]:.3f}, FAKE={proba_all[1]:.3f}\n")


if __name__ == "__main__":
    main()
