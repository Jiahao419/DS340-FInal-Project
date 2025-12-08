import pandas as pd

from config import PROCESSED_DIR, ensure_dirs


def main():
    ensure_dirs()

    datasets = ["main", "covid", "welfake"]
    splits = ["train", "val", "test"]

    out_dir = PROCESSED_DIR / "all"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        dfs = []
        for ds in datasets:
            path = PROCESSED_DIR / ds / f"{split}.csv"
            if not path.exists():
                print(f"Skip {ds}-{split}: {path} not found")
                continue
            print(f"Loading {ds}-{split} from {path}")
            df = pd.read_csv(path)
            df["source"] = ds
            dfs.append(df)

        if not dfs:
            print(f"No data found for split={split}, skip.")
            continue

        all_df = pd.concat(dfs, ignore_index=True)
        print(f"{split}: merged shape ->", all_df.shape)
        if "label_text" in all_df.columns:
            print("label distribution:")
            print(all_df["label_text"].value_counts(normalize=True))

        out_path = out_dir / f"{split}.csv"
        all_df.to_csv(out_path, index=False)
        print("Saved merged split to:", out_path)


if __name__ == "__main__":
    main()
