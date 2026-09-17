"""Regressions for defects found in the 2026-08-20 audit.

Each test pins the behaviour of one fix. None of these defects had corrupted a
stored result: they were either unreachable under the shipped config or needed a
crash at a specific instant. The tests exist so a future change cannot quietly
reintroduce them.

Defect 7 (an acceptance-suite timeout inside the figures stage left a stale
claim-status file behind) has no test here any more: defect 7's code was retired
on 2026-09-05 with the figures stage (design §17; docs/claims_archive_2026-09-05.md).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hnn2.result_io import _npz as save_npz
from hnn2.config import ExperimentConfig
from hnn2.metrics.bins import BinSpec, DroppedMassError
from hnn2.metrics.entropy import scores_from_counts
from hnn2.model.init import build_params
from hnn2.rng import STREAMS, stream, stream_torch_seed


# --- entropy: an unmeasurable histogram is not a perfect score --------------

def test_empty_histogram_scores_nan_not_maximal_concentration():
    scores = scores_from_counts(np.zeros(120))
    # 1.0 would rank a measurement that never happened as the most concentrated
    # result possible, which is how it would win a comparison it never entered.
    assert math.isnan(scores.S)
    assert math.isnan(scores.S_prime)


def test_nonempty_histogram_still_scores_normally():
    counts = np.zeros(4)
    counts[0] = 10.0
    scores = scores_from_counts(counts)
    assert scores.S == pytest.approx(1.0)
    assert not math.isnan(scores.S_prime)


# --- BinSpec: a range that is not a whole number of bins wide ---------------

def test_indivisible_binspec_is_rejected():
    # 1.0 / 0.3 is 3.33 bins: the top edge would land at 0.9, so the spec would
    # silently bin a narrower range than it advertises.
    with pytest.raises(ValueError, match="whole number of bins"):
        BinSpec(0.0, 1.0, 0.3)


def test_dropped_mass_message_quotes_the_edges_actually_used():
    spec = BinSpec(0.0, 1.0, 0.1)
    with pytest.raises(DroppedMassError) as excinfo:
        spec.apply(np.array([0.5, 1.5]))
    edges = spec.edges
    assert f"[{edges[0]}, {edges[-1]}]" in str(excinfo.value)


# --- init: a config asking for more than one I unit must not run silently ---

def test_multi_inhibitory_config_fails_loudly():
    config = ExperimentConfig(n_inhibitory=4)
    with pytest.raises(NotImplementedError, match="n_inhibitory=4"):
        build_params("rec", config, seed_index=0, n_input=100,
                     device=torch.device("cpu"))


def test_single_inhibitory_config_still_builds():
    params = build_params("rec", ExperimentConfig(), seed_index=0, n_input=100,
                          device=torch.device("cpu"))
    assert params.n_inhibitory == 1


# --- save_npz: a crash mid-write must not leave a readable-looking file -----

def test_save_npz_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "artifact.npz"
    save_npz(path, {"a": np.arange(4)})
    assert [p.name for p in tmp_path.iterdir()] == ["artifact.npz"]


def test_save_npz_overwrite_is_atomic(tmp_path):
    path = tmp_path / "artifact.npz"
    save_npz(path, {"a": np.arange(4)})
    save_npz(path, {"a": np.arange(2)})
    arrays = np.load(path, allow_pickle=False)
    assert arrays["a"].tolist() == [0, 1]


def test_save_npz_crash_mid_write_leaves_the_previous_file_intact(tmp_path,
                                                                  monkeypatch):
    path = tmp_path / "artifact.npz"
    save_npz(path, {"a": np.arange(4)})

    # Fail after some bytes have been written, which is the case that used to
    # leave a truncated file sitting at the target path. Callers guard on
    # path.exists(), so such a file would never be rewritten and every later
    # read of it would blow up on a corrupt zip.
    import hnn2.result_io as artifacts

    real_savez = artifacts.np.savez_compressed

    def savez_then_die(handle, **kwargs):
        handle.write(b"PK\x03\x04 truncated")
        raise OSError("disk full")

    monkeypatch.setattr(artifacts.np, "savez_compressed", savez_then_die)
    with pytest.raises(OSError):
        save_npz(path, {"a": np.arange(9)})
    monkeypatch.setattr(artifacts.np, "savez_compressed", real_savez)

    arrays = np.load(path, allow_pickle=False)
    assert arrays["a"].tolist() == [0, 1, 2, 3]
    assert [p.name for p in tmp_path.iterdir()] == ["artifact.npz"]


# --- RNG: streams append without disturbing the existing assignments --------

def test_stream_keys_are_unique_and_never_renumbered():
    # Renumbering an existing key silently changes every draw derived from it.
    baseline = {
        "init/w_EIn": 0, "init/w_IIn": 1, "shuffle": 2,
        "readout/init": 3, "readout/shuffle": 4,
        "mlp/init": 5, "mlp/shuffle": 6, "mlp/readout_init": 7, "umap": 8,
    }
    for name, key in baseline.items():
        assert STREAMS[name] == key
    assert len(set(STREAMS.values())) == len(STREAMS)




# --- Existing fixture lookup still follows the active generation ---------


