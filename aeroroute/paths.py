from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/bts_marketing"
PROCESSED_DIR = ROOT / "data/processed"
ACCESS_DIR = PROCESSED_DIR / "access"
