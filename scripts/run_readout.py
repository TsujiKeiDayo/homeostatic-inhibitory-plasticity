"""Fit only a new linear readout on explicitly chosen saved features."""

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
from hnn2.config import experiment_config_from_json
from hnn2.postprocess import readout_only
import json

from experiment_settings import (
    READOUT_SOURCE as SOURCE, READOUT_OUTPUT as OUTPUT, READOUT_LR,
    READOUT_EPOCHS, DEVICE, TORCH_THREADS, SETTINGS_PATH, SAVE_READOUT_HISTORY,
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(TORCH_THREADS)

    # Keep the saved experiment conditions and replace only readout settings.
    previous = json.loads((SOURCE / "conditions.json").read_text(encoding="utf-8"))
    config = experiment_config_from_json(json.dumps(previous["config"]))
    config = replace(config, readout_lr=READOUT_LR, readout_epochs=READOUT_EPOCHS)

    result = readout_only(
        SOURCE, config, OUTPUT, device=DEVICE, script_path=__file__,
        settings_path=SETTINGS_PATH, save_readout_history=SAVE_READOUT_HISTORY,
    )
    print(f"Final validation accuracy: {result.final_val_accuracy}; saved: {OUTPUT}")


if __name__ == "__main__":
    main()
