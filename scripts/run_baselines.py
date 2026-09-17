"""Run initialized sparse, raw, and frozen-MLP baselines with explicit seeds."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
from hnn2.workflows import run_baselines

from experiment_settings import (
    DATASET, BASELINES_OUTPUT as OUTPUT, DEVICE, TORCH_THREADS, SEEDS, MODELS,
    CONFIG, MLP, ANALYSIS, BASELINE_REPRESENTATIONS, SETTINGS_PATH,
    SAVING,
)


def main():
    torch.set_num_threads(TORCH_THREADS)
    states = run_baselines(
        DATASET, MODELS, SEEDS, CONFIG, OUTPUT,
        representations=BASELINE_REPRESENTATIONS, mlp=MLP, analysis=ANALYSIS,
        device=DEVICE, saving=SAVING, script_path=__file__, settings_path=SETTINGS_PATH,
    )
    print(states[["representation", "seed_index", "state", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
