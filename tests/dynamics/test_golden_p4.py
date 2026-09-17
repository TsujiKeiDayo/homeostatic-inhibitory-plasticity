"""Golden numerics for the whole adaptation / readout / MLP path (P4-0, 2026-09-06).

Small deterministic cases recorded on CPU from main at the close of P3
(commit 40849de, generation 8f962b529117 unchanged). Every later change to the
science core, the readout or the MLP must reproduce these arrays bit for bit
under the DEFAULT config — that is the proof the B-style hash rule demands when a
config field is added with a legacy value (CLAUDE.md, "config と hash の規則").

Recording (only when the science deliberately changes, recorded in PROGRESS):

    .venv/Scripts/python tests/dynamics/test_golden_p4.py --record

The cases are defined here, next to the assertions, so the fixture and the test
cannot drift apart; ``tests/fixtures/golden_settle.npz`` (settling only) predates
this file and keeps its own test in test_batched_core.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hnn2.adapt import run_adaptation
from hnn2.config import ExperimentConfig, RunSpec
from hnn2.mlp import train_mlp_e2e
from hnn2.readout import train_readout

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden_p4.npz"
DEVICE = torch.device("cpu")

# Small enough to run in seconds, large enough that every code path is exercised
# (two epochs: the first and the final probe are distinct; batches of 8 over 32
# samples: four homeostatic updates per epoch).
ADAPT_CONFIG = ExperimentConfig(n_excitatory=16, n_epochs=2, batch_size=8, n_stabilise=2,
                                settle_steps=20)
ADAPT_SPECS = {
    "rec": RunSpec(model_code="rec", targ=0.20, eta=1e-2, seed_index=0),
    "ff": RunSpec(model_code="ff", targ=0.35, eta=1e-3, seed_index=1),
    "thresh": RunSpec(model_code="thresh", targ=0.50, eta=0.1, seed_index=2),
}
READOUT_CONFIG = ExperimentConfig(n_excitatory=16, readout_epochs=2, batch_size=8)
N_INPUT, N_SAMPLES, N_CLASSES = 12, 32, 4


def _splits() -> dict[str, torch.Tensor]:
    """Non-negative pixel-like inputs, (n_input, n_samples) per split, fixed seed."""
    rng = np.random.default_rng(20260906)
    return {split: torch.as_tensor(rng.random((N_INPUT, N_SAMPLES), dtype=np.float32) ** 2)
            for split in ("train", "val", "test")}


def compute() -> dict[str, np.ndarray]:
    """Every golden array, keyed ``<case>_<name>``."""
    out: dict[str, np.ndarray] = {}
    splits = _splits()
    for model, spec in ADAPT_SPECS.items():
        result = run_adaptation(spec, ADAPT_CONFIG, splits, DEVICE)
        arrays = result.summaries_arrays()
        for key in ("param_trajectory", "monitor_rate_mean", "monitor_rate_wmean",
                    "monitor_composite", "monitor_rate_mean_train",
                    "probe_ep00_r_pop", "probe_ep01_r_pop", "probe_ep01_hist_pop",
                    "probe_ep01_e_input_life", "probe_ep01_i_input_life"):
            out[f"{model}_{key}"] = np.asarray(arrays[key])
        out[f"{model}_o_E_final"] = result.analysis_activity[ADAPT_CONFIG.n_epochs - 1]
        for name, tensor in result.params_trained.to_arrays().items():
            out[f"{model}_trained_{name}"] = tensor

    rng = np.random.default_rng(6092026)
    features = {f"h_{s}": rng.random((N_SAMPLES, 16), dtype=np.float32)
                for s in ("train", "val", "test")}
    labels = {f"y_{s}": rng.integers(0, N_CLASSES, N_SAMPLES) for s in ("train", "val", "test")}
    readout = train_readout(features, labels, READOUT_CONFIG, init_seed=11, shuffle_seed=13,
                            device=DEVICE)
    out["readout_val_accuracy"] = np.asarray(readout.history.val_accuracy)
    out["readout_batch_loss"] = np.asarray(readout.history.batch_loss)
    out["readout_cumulative_l1"] = np.asarray(readout.history.cumulative_l1)
    out["readout_final"] = np.asarray([readout.final_val_accuracy, readout.final_test_accuracy])

    x_train = rng.random((N_SAMPLES, N_INPUT), dtype=np.float32)
    mlp = train_mlp_e2e(x_train, labels["y_train"], READOUT_CONFIG, init_seed=17, shuffle_seed=19,
                        device=DEVICE)
    out["mlp_hidden_weight"] = mlp.hidden.weight.detach().numpy()
    out["mlp_readout_weight"] = mlp.readout.weight.detach().numpy()
    return out


def record() -> None:
    arrays = compute()
    np.savez_compressed(GOLDEN, **arrays)
    print(f"wrote {GOLDEN} ({len(arrays)} arrays)")


@pytest.fixture(scope="module")
def computed() -> dict[str, np.ndarray]:
    return compute()


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    if not GOLDEN.exists():
        pytest.fail(f"{GOLDEN} missing — record it with `python {Path(__file__).name} --record`")
    with np.load(GOLDEN) as npz:
        return {key: npz[key] for key in npz.files}


def test_golden_covers_every_computed_array(computed, golden):
    assert set(computed) == set(golden)


@pytest.mark.parametrize("model", sorted(ADAPT_SPECS))
def test_adaptation_is_bit_identical(model, computed, golden):
    keys = [k for k in golden if k.startswith(f"{model}_")]
    assert keys
    for key in keys:
        assert np.array_equal(computed[key], golden[key]), key


def test_readout_and_mlp_are_bit_identical(computed, golden):
    for key in [k for k in golden if k.startswith(("readout_", "mlp_"))]:
        assert np.array_equal(computed[key], golden[key]), key


if __name__ == "__main__":
    if "--record" in sys.argv:
        record()
    else:
        print(__doc__)
