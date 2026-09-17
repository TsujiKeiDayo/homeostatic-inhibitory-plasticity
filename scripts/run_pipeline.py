"""Run HP selection, selected experiments, and baselines from prepared input."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
from hnn2.data import load_dataset
from hnn2.workflows import (
    run_sweep_selection, run_selected_experiments, run_baselines,
    baseline_population, save_expected_runs,
)
from experiment_settings import (
    DATASET, CONFIG, SWEEP_CONFIG, SWEEP_SPECS, RULE, DEVICE, TORCH_THREADS,
    BATCH_RUNS, SWEEP_OUTPUT, SELECTION_OUTPUT, EXPERIMENTS_OUTPUT,
    ANALYSIS, SETTINGS_PATH, SAVING,
    BASELINES_OUTPUT, BASELINE_REPRESENTATIONS, MLP,
)


def main():
    torch.set_num_threads(TORCH_THREADS)
    dataset = load_dataset(DATASET)

    # Preserve the expected baseline population even if an earlier stage fails.
    save_expected_runs(
        BASELINES_OUTPUT,
        baseline_population(
            BASELINES_OUTPUT, list(dict.fromkeys(s.model_code for s in SWEEP_SPECS)),
            sorted({s.seed_index for s in SWEEP_SPECS}), BASELINE_REPRESENTATIONS,
        ),
        config=CONFIG,
    )

    # Select eta for each model/target, then train the selected conditions.
    run_sweep_selection(
        dataset, SWEEP_SPECS, SWEEP_CONFIG, RULE, SWEEP_OUTPUT, SELECTION_OUTPUT,
        training_config=CONFIG, batch_runs=BATCH_RUNS, device=DEVICE,
        script_path=__file__, settings_path=SETTINGS_PATH,
    )
    states = run_selected_experiments(
        SELECTION_OUTPUT, dataset, CONFIG, EXPERIMENTS_OUTPUT,
        batch_runs=BATCH_RUNS, device=DEVICE, analysis=ANALYSIS, saving=SAVING,
        script_path=__file__, settings_path=SETTINGS_PATH,
    )

    # Baselines use the models and seeds actually inherited from the HP selection.
    baselines = run_baselines(
        dataset, states.model_code.drop_duplicates().tolist(),
        states.seed_index.drop_duplicates().tolist(), CONFIG, BASELINES_OUTPUT,
        representations=BASELINE_REPRESENTATIONS, mlp=MLP, analysis=ANALYSIS,
        device=DEVICE, saving=SAVING, script_path=__file__, settings_path=SETTINGS_PATH,
    )

    print(states[["model_code", "targ", "eta", "seed_index", "state", "reason"]].to_string(index=False))
    print(baselines[["representation", "seed_index", "state", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
