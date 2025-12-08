import pandas as pd
from pathlib import Path

from config import RAW_DIR, PROCESSED_DIR, ensure_dirs


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = " ".join(text.split())
    return text


def main():
    ensure_dirs()

    covid_raw_dir = RAW_DIR / "covid"
    out_dir = PROCESSED_DIR / "covid"

    INPUT_FILES = {
        "train": "Constraint_Train.csv",
        "val": "Constraint_Val.csv",
        "test": "Constraint_Test.csv",
    }

    for split, fname in INPUT_FILES.items():
        path = covid_raw_dir / fname
        if not path.exists():
            print(f"Skip {split}: file not found -> {path}")
            continue

        print(f"\nReading {split} from:", path)
        df = pd.read_csv(path)
        print("raw shape:", df.shape)
        print("columns:", df.columns)

        # ---------- Labeled splits (train / val) ----------
        if "label" in df.columns:
            if "tweet" not in df.columns:
                raise ValueError(f"Cannot find 'tweet' column in {fname}, please check.")

            df = df[["tweet", "label"]].copy()

            # Normalize labels: 1 = FAKE, 0 = REAL
            df["label"] = df["label"].str.lower().map({"fake": 1, "real": 0})
            if df["label"].isna().any():
                raise ValueError(f"{fname} contains labels other than fake/real, please check.")

            df["label"] = df["label"].astype(int)
            df["label_text"] = df["label"].map({1: "FAKE", 0: "REAL"})

            print("Label distribution:")
            print(df["label_text"].value_counts(normalize=True))

        # ---------- Unlabeled split (test) ----------
        else:
            if "tweet" not in df.columns:
                raise ValueError(f"Cannot find 'tweet' column in {fname}, please check.")
            keep_cols = [c for c in df.columns if c in ["id", "tweet"]]
            df = df[keep_cols].copy()

        # ---------- Common cleaning ----------
        df["clean_text"] = df["tweet"].astype(str).apply(basic_clean)

        before = len(df)
        df = df[df["clean_text"].str.len() > 5].copy()
        after = len(df)
        print(f"Removed {before - after} very short/empty rows.")

        before = len(df)
        subset_cols = ["clean_text"] + (["label"] if "label" in df.columns else [])
        df = df.drop_duplicates(subset=subset_cols).copy()
        after = len(df)
        print(f"Removed {before - after} duplicate rows.")

        df = df.reset_index(drop=True)
        df["id"] = df.index

        out_path = out_dir / f"{split}.csv"
        df.to_csv(out_path, index=False)
        print("Saved cleaned split to:", out_path)


if __name__ == "__main__":
    main()
