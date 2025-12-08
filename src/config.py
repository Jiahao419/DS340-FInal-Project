from pathlib import Path

# Project root directory: assuming config.py is under src/
ROOT = Path(__file__).resolve().parents[1]

# Data directory
DATA_DIR = ROOT / "data"

# Raw datasets directory (covid / welfake are under this)
RAW_DIR = DATA_DIR

# Output directory (all processed + features go here)
OUT_DIR = ROOT / "output"

# Processed train/val/test
PROCESSED_DIR = OUT_DIR / "processed"

# Feature-enhanced versions
FEATURES_DIR = OUT_DIR / "features"

# Supported datasets
DATASETS = ["main", "covid", "welfake", "all"]


def ensure_dirs():
    """
    Create necessary output directories.
    """
    for base in [PROCESSED_DIR, FEATURES_DIR]:
        for ds in DATASETS:
            (base / ds).mkdir(parents=True, exist_ok=True)
