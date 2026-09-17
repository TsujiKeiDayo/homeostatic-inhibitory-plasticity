"""G1/G2 gates of the batched engine (design v1.4 D20; SPEC V12).

G1  ``run_adaptation_batch`` over mixed (targ, eta, seed) specs reproduces
    ``run_adaptation`` run by run: monitor records, parameter trajectory,
    final weights and analysis activity, up to batched-matmul rounding.
G2  prefix property: the first T epochs of a longer run are the T-epoch run
    (bit for bit on the sequential path), which is what lets the sweep read
    every n_epochs off one horizon-length run.
"""

import numpy as np
import pytest
import torch

from hnn2.adapt import run_adaptation
from hnn2.config import ExperimentConfig, RunSpec
from hnn2.hp.engine import chunk, run_adaptation_batch

DEVICE = torch.device("cpu")
# Small everything: the equivalence is structural, not statistical.
CONFIG = ExperimentConfig(n_excitatory=48, settle_steps=15, n_stabilise=2,
                          batch_size=8, n_epochs=3, eta_decay=0.9, eta_decay_start_epoch=1)


def _splits(n_input=20, seed=7):
    rng = np.random.default_rng(seed)
    make = lambda n: torch.as_tensor(rng.random((n_input, n)), dtype=torch.float32)
    return {"train": make(32), "val": make(16), "test": make(16)}


# Etas within each variant's realistic range (its selected eta, a third, three
# times). Far above it the ReLU gates flip on rounding differences alone, which
# a batched matmul and a single matmul do not share; see the last test.
ETAS = {"rec": 1e-3, "ff": 1e-4, "thresh": 1e-1}


def _specs(model):
    eta = ETAS[model]
    return [RunSpec(model, 0.2, eta, 0), RunSpec(model, 0.35, eta / 3, 1), RunSpec(model, 0.2, 3 * eta, 2)]


def _records(result, key):
    return np.array([r[key] for r in result.monitor_records])


@pytest.mark.parametrize("model_code", ["rec", "ff", "thresh"])
def test_batched_matches_sequential_run_by_run(model_code):
    splits = _splits()
    specs = _specs(model_code)
    batched = run_adaptation_batch(specs, CONFIG, splits, DEVICE)
    assert len(batched) == len(specs)
    for spec, got in zip(specs, batched):
        ref = run_adaptation(spec, CONFIG, splits, DEVICE)
        for key in ("l_mean", "rate_var", "rate_mean", "rate_wmean", "composite",
                    "rate_mean_train", "rate_wmean_train"):
            # atol: l_mean = (rate - targ)^2 cancels catastrophically near the
            # target, so a 1e-5 relative wobble in rate shows up as 1e-3 in it.
            np.testing.assert_allclose(_records(got, key), _records(ref, key),
                                       rtol=1e-4, atol=1e-6, err_msg=key)
        assert _records(got, "eta").tolist() == _records(ref, "eta").tolist()
        assert [r["epoch"] for r in got.monitor_records] == [r["epoch"] for r in ref.monitor_records]
        assert got.trajectory_index == ref.trajectory_index
        np.testing.assert_allclose(np.stack(got.param_trajectory), np.stack(ref.param_trajectory),
                                   rtol=1e-5, atol=1e-7)
        for name in ("w_EIn", "theta", "w_EI", "w_IE", "w_IIn"):
            a, b = getattr(got.params_trained, name), getattr(ref.params_trained, name)
            assert (a is None) == (b is None)
            if a is not None:
                np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-7)
        # Initial weights come from the same streams: bit-identical.
        assert torch.equal(got.params_initial.w_EIn, ref.params_initial.w_EIn)
        assert set(got.analysis_probes) == set(ref.analysis_probes) == {0, CONFIG.n_epochs - 1}
        for epoch in got.analysis_probes:
            np.testing.assert_allclose(got.analysis_activity[epoch], ref.analysis_activity[epoch],
                                       rtol=1e-4, atol=1e-6)
            for key, value in ref.analysis_probes[epoch].items():
                np.testing.assert_allclose(got.analysis_probes[epoch][key], value,
                                           rtol=1e-4, atol=1e-6, err_msg=key)


def test_sweep_mode_skips_trajectory_and_probes():
    results = run_adaptation_batch(_specs("rec"), CONFIG, _splits(), DEVICE,
                                   keep_trajectory=False, analysis_probes=False)
    for result in results:
        assert result.param_trajectory == [] and result.trajectory_index == []
        assert result.analysis_probes == {} and result.analysis_activity == {}
        assert len(result.monitor_records) == CONFIG.n_epochs


def test_batch_rejects_mixed_variants():
    with pytest.raises(ValueError):
        run_adaptation_batch([RunSpec("rec", 0.2, 1e-3, 0), RunSpec("ff", 0.2, 1e-3, 0)],
                             CONFIG, _splits(), DEVICE)


def test_batch_of_one_equals_the_single_run():
    spec = RunSpec("ff", 0.35, 1e-4, 0)
    splits = _splits()
    got = run_adaptation_batch([spec], CONFIG, splits, DEVICE)[0]
    ref = run_adaptation(spec, CONFIG, splits, DEVICE)
    np.testing.assert_allclose(_records(got, "rate_mean"), _records(ref, "rate_mean"), rtol=1e-5)


def test_prefix_property_on_the_sequential_path():
    """G2: epochs 0..T-1 of an n_epochs=T+k run equal the n_epochs=T run exactly."""
    splits = _splits()
    spec = RunSpec("rec", 0.2, 1e-3, 0)
    short = run_adaptation(spec, CONFIG, splits, DEVICE)
    longer = run_adaptation(spec, ExperimentConfig(**{**CONFIG.__dict__, "n_epochs": 6}),
                            splits, DEVICE)
    n = CONFIG.n_epochs
    for key in ("l_mean", "rate_var", "rate_mean", "rate_wmean", "rate_mean_train",
                "rate_wmean_train", "eta"):
        assert _records(short, key).tolist() == _records(longer, key)[:n].tolist(), key
    assert short.trajectory_index == longer.trajectory_index[:len(short.trajectory_index)]
    assert np.array_equal(np.stack(short.param_trajectory),
                          np.stack(longer.param_trajectory)[:len(short.param_trajectory)])


def test_large_eta_differences_are_gate_flips_not_structure():
    """At an eta far above ff's range (1e-2, a hundred times its selected
    value) a handful of units sit on the ReLU threshold and the two matmul
    kernels round them to different sides: the same happens between R=1 and
    the sequential path, so it is rounding amplification, not batching. The
    disagreement stays confined to a minority of units and modest in size."""
    splits = _splits()
    spec = RunSpec("ff", 0.2, 1e-2, 2)
    ref = np.stack(run_adaptation(spec, CONFIG, splits, DEVICE).param_trajectory)
    single = np.stack(run_adaptation_batch([spec], CONFIG, splits, DEVICE)[0].param_trajectory)
    stacked = run_adaptation_batch(_specs("ff")[:2] + [spec], CONFIG, splits, DEVICE)[2]
    stacked = np.stack(stacked.param_trajectory)
    for got in (single, stacked):
        rel = np.abs(got - ref) / (np.abs(ref) + 1e-12)
        assert rel.max() < 0.05
        assert (rel.max(axis=0) > 1e-5).mean() < 0.25
    # R=1 and R=3 see the same rounding: batching adds nothing on top.
    np.testing.assert_allclose(single, stacked, rtol=1e-5, atol=1e-7)


def test_chunk():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk([], 3) == []
    with pytest.raises(ValueError):
        chunk([1], 0)
