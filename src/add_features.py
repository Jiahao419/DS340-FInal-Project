import argparse

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import PROCESSED_DIR, FEATURES_DIR, ensure_dirs, DATASETS


def add_advanced_features(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    # 1. Basic length features
    df["num_chars"] = df[text_col].str.len()
    df["num_words"] = df[text_col].str.split().str.len()
    df["avg_word_len"] = df["num_chars"] / df["num_words"].replace(0, 1)

    # 2. Punctuation features
    df["num_exclamation"] = df[text_col].str.count("!")
    df["num_question"] = df[text_col].str.count(r"\?")

    # 3. Lexical richness
    tokens = df[text_col].str.split()
    df["unique_words"] = tokens.apply(lambda x: len(set(x)) if isinstance(x, list) else 0)
    df["lexical_richness"] = df["unique_words"] / df["num_words"].replace(0, 1)

    # 4. Stopword ratio
    stopwords = ENGLISH_STOP_WORDS

    def count_stopwords(tok_list):
        if not isinstance(tok_list, list):
            return 0
        return sum(1 for w in tok_list if w in stopwords)

    df["num_stopwords"] = tokens.apply(count_stopwords)
    df["stopword_ratio"] = df["num_stopwords"] / df["num_words"].replace(0, 1)

    # 5. Digits and URLs
    df["num_digits"] = df[text_col].str.count(r"\d")
    df["digit_ratio"] = df["num_digits"] / df["num_chars"].replace(0, 1)

    url_pattern = r"http[s]?://|www\."
    df["num_urls"] = df[text_col].str.count(url_pattern)

    # 6. Optional: year/month if available, otherwise set to NA
    date_col = None
    for c in ["date", "publish_date"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
    else:
        df["year"] = pd.NA
        df["month"] = pd.NA

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="main",
        choices=DATASETS,
        help="Which dataset to add features for (main/covid/welfake/all)",
    )
    args = parser.parse_args()

    ensure_dirs()

    in_dir = PROCESSED_DIR / args.dataset
    out_dir = FEATURES_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        path = in_dir / f"{split}.csv"
        if not path.exists():
            print(f"Skip {split}: {path} not found")
            continue

        print("Reading:", path)
        df = pd.read_csv(path)

        if "clean_text" not in df.columns:
            raise ValueError(f"{path} does not contain clean_text column. Check the previous prepare script.")

        df = add_advanced_features(df, text_col="clean_text")

        out_path = out_dir / f"{split}_fe.csv"
        df.to_csv(out_path, index=False)
        print("Saved with features ->", out_path)


if __name__ == "__main__":
    main()
