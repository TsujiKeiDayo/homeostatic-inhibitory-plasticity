"""Explicit MNIST preparation; this is the only entry that may download data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hnn2.data import create_dataset

from experiment_settings import RAW_CACHE, DATASET_OUTPUT as OUTPUT, DATASET_PARAMS as PARAMS


if __name__ == "__main__":
    print(create_dataset(OUTPUT, PARAMS, RAW_CACHE))
