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

    # Note: fake / true are under data/welfake/
    fake_path = RAW_DIR / "welfake" / "fake.csv"
    true_path = RAW_DIR / "welfake" / "true.csv"

    print("Fake path:", fake_path)
    print("True path:", true_path)

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    print("fake shape:", fake.shape)
    print("true shape:", true.shape)

    # ---------- Assign labels ----------
    fake["label"] = 1
    fake["label_text"] = "FAKE"

    true["label"] = 0
    true["label_text"] = "REAL"

    df = pd.concat([fake, true], ignore_index=True)
    print("combined shape:", df.shape)
    print("label counts:")
    print(df["label_text"].value_counts())

    # ---------- Build clean_text ----------
    if not {"title", "text"}.issubset(df.columns):
        raise ValueError("The main dataset must contain 'title' and 'text' columns.")

    df["clean_text"] = (
        df["title"].fillna("").astype(str)
        + "\n\n"
        + df["text"].fillna("").astype(str)
    )
    df["clean_text"] = df["clean_text"].apply(basic_clean)

    # Drop very short/empty texts
    before = len(df)
    df = df[df["clean_text"].str.len() > 20].copy()
    after = len(df)
    print(f"Removed {before - after} very short/empty rows.")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["clean_text"]).copy()
    after = len(df)
    print(f"Removed {before - after} duplicate rows.")

    df = df.reset_index(drop=True)
    df["id"] = df.index

    # ---------- Split into train / val / test ----------
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

    def show_dist(name, d):
        print(f"\n{name} label distribution:")
        print(d["label_text"].value_counts(normalize=True))

    print("Final sizes -> train:", train_df.shape,
          "val:", val_df.shape,
          "test:", test_df.shape)

    show_dist("Train", train_df)
    show_dist("Val", val_df)
    show_dist("Test", test_df)

    out_dir = PROCESSED_DIR / "main"
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("\nSaved main dataset to:", out_dir)


if __name__ == "__main__":
    main()
