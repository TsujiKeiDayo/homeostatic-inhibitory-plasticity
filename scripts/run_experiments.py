"""Train all saved HP model/target cells at their selected eta and original seeds."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
from hnn2.workflows import run_selected_experiments

from experiment_settings import (
    DATASET, EXPERIMENTS_OUTPUT as OUTPUT, DEVICE, TORCH_THREADS, BATCH_RUNS,
    CONFIG, SELECTION_OUTPUT as SOURCE, ANALYSIS, SETTINGS_PATH,
    SAVING,
)


def main():
    torch.set_num_threads(TORCH_THREADS)
    states = run_selected_experiments(
        SOURCE, DATASET, CONFIG, OUTPUT, batch_runs=BATCH_RUNS,
        device=DEVICE, analysis=ANALYSIS, saving=SAVING,
        script_path=__file__, settings_path=SETTINGS_PATH,
    )
    print(states[["model_code", "targ", "eta", "seed_index", "state", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
