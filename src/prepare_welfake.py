import pandas as pd
from sklearn.model_selection import train_test_split

from config import RAW_DIR, PROCESSED_DIR, ensure_dirs


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = " ".join(text.split())
    return text


def main():
    ensure_dirs()

    data_dir = RAW_DIR / "welfake"
    raw_path = data_dir / "WELFake_Dataset.csv"

    print("Reading:", raw_path)
    df = pd.read_csv(raw_path)
    print("raw shape:", df.shape)
    print("columns:", df.columns)

    # 1. Select required columns
    if "title" not in df.columns or "text" not in df.columns or "label" not in df.columns:
        raise ValueError("WELFake dataset must contain 'title', 'text', and 'label' columns.")

    df = df[["title", "text", "label"]].copy()

    # 2. Normalize label semantics: 1 = FAKE, 0 = REAL
    # Original: 0=fake, 1=real -> flip to 1=FAKE, 0=REAL
    df["label"] = df["label"].astype(int)
    df["label"] = df["label"].apply(lambda v: 1 if v == 0 else 0)
    df["label_text"] = df["label"].map({1: "FAKE", 0: "REAL"})

    print("Label distribution after remap:")
    print(df["label_text"].value_counts(normalize=True))

    # 3. Build clean_text
    df["clean_text"] = (
        df["title"].fillna("").astype(str)
        + "\n\n"
        + df["text"].fillna("").astype(str)
    )
    df["clean_text"] = df["clean_text"].apply(basic_clean)

    before = len(df)
    df = df[df["clean_text"].str.len() > 20].copy()
    after = len(df)
    print(f"Removed {before - after} very short/empty rows.")

    before = len(df)
    df = df.drop_duplicates(subset=["clean_text"]).copy()
    after = len(df)
    print(f"Removed {before - after} duplicate rows.")

    df = df.reset_index(drop=True)
    df["id"] = df.index

    print("clean shape:", df.shape)

    # 4. Split into train/val/test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    print("Final sizes -> train:", train_df.shape,
          "val:", val_df.shape,
          "test:", test_df.shape)

    def show_dist(name, d):
        print(f"\n{name} label distribution:")
        print(d["label_text"].value_counts(normalize=True))

    show_dist("Train", train_df)
    show_dist("Val", val_df)
    show_dist("Test", test_df)

    out_dir = PROCESSED_DIR / "welfake"
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("\nSaved WELFake dataset to:", out_dir)


if __name__ == "__main__":
    main()
