"""Run explicit HP candidates and select eta with the stability-filter rule."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
from hnn2.workflows import run_sweep_selection

from experiment_settings import (
    DATASET, SWEEP_OUTPUT as OUTPUT, SELECTION_OUTPUT, DEVICE, TORCH_THREADS,
    BATCH_RUNS, SWEEP_CONFIG as CONFIG, SWEEP_SPECS as SPECS, RULE, SETTINGS_PATH,
    CONFIG as TRAINING_CONFIG,
)


def main():
    torch.set_num_threads(TORCH_THREADS)
    selection = run_sweep_selection(
        DATASET, SPECS, CONFIG, RULE, OUTPUT, SELECTION_OUTPUT,
        training_config=TRAINING_CONFIG, batch_runs=BATCH_RUNS, device=DEVICE,
        script_path=__file__, settings_path=SETTINGS_PATH,
    )
    print(f"Selected {selection['n_cells']} model/target cells: {SELECTION_OUTPUT}")


if __name__ == "__main__":
    main()
