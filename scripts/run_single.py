"""Run and save one condition configured in experiment_settings.py."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hnn2.single import run_single
from hnn2.result_io import read_single, save_single
import torch

from experiment_settings import (
    DATASET, SINGLE_OUTPUT as OUTPUT, DEVICE, TORCH_THREADS, SINGLE_SPEC as SPEC,
    CONFIG, ANALYSIS, SETTINGS_PATH, SAVING,
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Choose a new output directory: {OUTPUT}")
    torch.set_num_threads(TORCH_THREADS)

    result = run_single(DATASET, SPEC, CONFIG, analysis=ANALYSIS, device=DEVICE)
    directory = save_single(
        result, OUTPUT, script_path=__file__, settings_path=SETTINGS_PATH, saving=SAVING
    )

    conditions, arrays, tables = read_single(directory)
    print(f"Saved: {directory}")
    print(tables["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
