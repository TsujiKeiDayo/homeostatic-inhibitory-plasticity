"""Leading run dimension in the model core (design v1.4, D20; SPEC V12).

Two guarantees:

1. The single-network path is BIT-IDENTICAL to golden values recorded before
   the run dimension existed (tests/fixtures/golden_settle.npz, CPU). Any
   change to dynamics/plasticity/simulate that alters a 2-D kernel fails here.
2. R stacked networks settle, warm-start, encode and update exactly as R
   separate calls would, up to batched-matmul rounding (rel 1e-5).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from hnn2.config import ExperimentConfig
from hnn2.model.init import build_params
from hnn2.model.params import ModelParams, zero_state
from hnn2.model.plasticity import update
from hnn2.simulate import encode, settle, warm_start_state

CONFIG = ExperimentConfig(settle_steps=50)
DEVICE = torch.device("cpu")
GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden_settle.npz"
MODELS = ("rec", "ff", "thresh")


def _x(n_input=100, batch=8, seed=123) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.random((n_input, batch)), dtype=torch.float32, device=DEVICE)


def _close(a: torch.Tensor, b: torch.Tensor) -> None:
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 1. the single-network path has not moved
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_code", MODELS)
def test_single_path_is_bit_identical_to_golden(model_code):
    golden = np.load(GOLDEN)
    params = build_params(model_code, CONFIG, seed_index=0, n_input=100, device=DEVICE)
    state = settle(params, _x(), CONFIG)
    assert np.array_equal(state.o_E.numpy(), golden[f"{model_code}_o_E"])
    assert np.array_equal(state.v_E.numpy(), golden[f"{model_code}_v_E"])
    pre = state.v_I if model_code != "thresh" else None
    new = update(params, state.o_E, pre, targ=0.3, eta=1e-2)
    adapted = new.w_EI if model_code != "thresh" else new.theta
    assert np.array_equal(adapted.numpy(), golden[f"{model_code}_updated"])


# ---------------------------------------------------------------------------
# 2. stacked networks behave like separate ones
# ---------------------------------------------------------------------------

def _singles(model_code, n=3):
    return [build_params(model_code, CONFIG, seed_index=s, n_input=100, device=DEVICE)
            for s in range(n)]


@pytest.mark.parametrize("model_code", MODELS)
def test_stack_and_select_round_trip(model_code):
    singles = _singles(model_code)
    stacked = ModelParams.stack(singles)
    assert stacked.batch_shape == (3,)
    assert stacked.n_excitatory == singles[0].n_excitatory
    assert stacked.n_input == singles[0].n_input
    assert stacked.n_inhibitory == singles[0].n_inhibitory
    for r, single in enumerate(singles):
        picked = stacked.select(r)
        assert picked.batch_shape == ()
        assert torch.equal(picked.w_EIn, single.w_EIn)
        assert torch.equal(picked.theta, single.theta)
        for name in ("w_EI", "w_IE", "w_IIn"):
            a, b = getattr(picked, name), getattr(single, name)
            assert (a is None) == (b is None)
            if a is not None:
                assert torch.equal(a, b)


@pytest.mark.parametrize("model_code", MODELS)
def test_stacked_settle_with_shared_input(model_code):
    singles = _singles(model_code)
    stacked = ModelParams.stack(singles)
    x = _x()
    batched = settle(stacked, x, CONFIG)
    assert batched.o_E.shape == (3, CONFIG.n_excitatory, 8)
    for r, single in enumerate(singles):
        ref = settle(single, x, CONFIG)
        _close(batched.o_E[r], ref.o_E)
        _close(batched.v_E[r], ref.v_E)
        if model_code != "thresh":
            _close(batched.v_I[r], ref.v_I)


@pytest.mark.parametrize("model_code", MODELS)
def test_stacked_settle_with_per_run_input(model_code):
    singles = _singles(model_code)
    stacked = ModelParams.stack(singles)
    xs = torch.stack([_x(seed=10 + r) for r in range(3)])   # (3, nIn, B)
    batched = settle(stacked, xs, CONFIG)
    for r, single in enumerate(singles):
        _close(batched.o_E[r], settle(single, xs[r], CONFIG).o_E)


def test_zero_state_carries_the_run_dimension():
    stacked = ModelParams.stack(_singles("rec"))
    state = zero_state(stacked, 5, DEVICE)
    assert state.v_E.shape == (3, CONFIG.n_excitatory, 5)
    assert state.v_I.shape == (3, 1, 5)
    thresh = ModelParams.stack(_singles("thresh"))
    assert zero_state(thresh, 5, DEVICE).v_I.shape == (3, 0, 5)


@pytest.mark.parametrize("model_code", MODELS)
def test_stacked_warm_start_and_encode(model_code):
    singles = _singles(model_code)
    stacked = ModelParams.stack(singles)
    x_train = _x(batch=32, seed=9)
    warm = warm_start_state(stacked, x_train, CONFIG)
    assert warm.v_E.shape == (3, CONFIG.n_excitatory, 1)
    out = encode(stacked, x_train[:, :4], CONFIG, warm)
    assert out.o_E.shape == (3, CONFIG.n_excitatory, 4)
    for r, single in enumerate(singles):
        warm_r = warm_start_state(single, x_train, CONFIG)
        _close(warm.v_E[r], warm_r.v_E)
        _close(out.o_E[r], encode(single, x_train[:, :4], CONFIG, warm_r).o_E)


@pytest.mark.parametrize("model_code", MODELS)
def test_stacked_update_uses_per_run_targ_and_eta(model_code):
    singles = _singles(model_code)
    stacked = ModelParams.stack(singles)
    x = _x()
    state = settle(stacked, x, CONFIG)
    targs = torch.tensor([0.1, 0.2, 0.3])
    etas = torch.tensor([1e-3, 3e-3, 1e-2])
    pre = state.v_I if model_code != "thresh" else None
    new = update(stacked, state.o_E, pre, targ=targs, eta=etas)
    for r, single in enumerate(singles):
        ref_state = settle(single, x, CONFIG)
        ref_pre = ref_state.v_I if model_code != "thresh" else None
        ref = update(single, ref_state.o_E, ref_pre, targ=float(targs[r]), eta=float(etas[r]))
        if model_code != "thresh":
            _close(new.w_EI[r], ref.w_EI)
        else:
            _close(new.theta[r], ref.theta)


def test_update_with_scalar_targ_on_stacked_params_broadcasts():
    stacked = ModelParams.stack(_singles("rec"))
    state = settle(stacked, _x(), CONFIG)
    new = update(stacked, state.o_E, state.v_I, targ=0.2, eta=1e-3)
    assert new.w_EI.shape == stacked.w_EI.shape
