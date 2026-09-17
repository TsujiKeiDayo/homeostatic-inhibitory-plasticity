"""T2 property tests: monotonicity, warm-start determinism, update direction."""

import numpy as np
import torch

from hnn2.config import ExperimentConfig
from hnn2.model.init import build_params
from hnn2.model.plasticity import eta_schedule, update
from hnn2.simulate import encode, settle, warm_start_state

CONFIG = ExperimentConfig()
DEVICE = torch.device("cpu")


def _batch(n_input=100, batch=8, seed=5):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.random((n_input, batch)), dtype=torch.float32, device=DEVICE)


def test_stronger_inhibition_never_increases_output():
    params = build_params("rec", CONFIG, seed_index=1, n_input=100, device=DEVICE)
    x = _batch()
    base = settle(params, x, CONFIG).o_E
    stronger = settle(params.with_(w_EI=params.w_EI * 2.0), x, CONFIG).o_E
    assert torch.all(stronger <= base + 1e-6)


def test_higher_threshold_never_increases_output():
    params = build_params("thresh", CONFIG, seed_index=1, n_input=100, device=DEVICE)
    x = _batch()
    base = settle(params, x, CONFIG).o_E
    raised = settle(params.with_(theta=params.theta + 0.5), x, CONFIG).o_E
    assert torch.all(raised <= base + 1e-6)


def test_warm_start_deterministic_and_used_by_encode():
    params = build_params("ff", CONFIG, seed_index=2, n_input=100, device=DEVICE)
    x_train = _batch(batch=32, seed=9)
    warm_a = warm_start_state(params, x_train, CONFIG)
    warm_b = warm_start_state(params, x_train, CONFIG)
    assert torch.equal(warm_a.v_E, warm_b.v_E)
    assert torch.equal(warm_a.v_I, warm_b.v_I)

    out = encode(params, x_train[:, :4], CONFIG, warm_a)
    assert out.o_E.shape == (CONFIG.n_excitatory, 4)


def test_update_direction_matches_error_sign_rec():
    params = build_params("rec", CONFIG, seed_index=0, n_input=100, device=DEVICE)
    nE, nI, B = CONFIG.n_excitatory, CONFIG.n_inhibitory, 4
    o_E = torch.zeros((nE, B))
    o_E[:10] = 2.0   # above target -> inhibition must strengthen
    pre = torch.ones((nI, B))
    new = update(params, o_E, pre, targ=0.5, eta=1e-2)
    assert torch.all(new.w_EI[:10] > params.w_EI[:10])
    assert torch.all(new.w_EI[10:] <= params.w_EI[10:])  # below target -> weaken


def test_update_clamps_at_zero():
    params = build_params("ff", CONFIG, seed_index=0, n_input=100, device=DEVICE)
    o_E = torch.zeros((CONFIG.n_excitatory, 4))          # far below target
    pre = torch.ones((CONFIG.n_inhibitory, 4))
    new = update(params, o_E, pre, targ=5.0, eta=1e3)    # huge depressive step
    assert torch.all(new.w_EI >= 0.0)
    assert torch.all(new.w_EI == 0.0)


def test_update_direction_thresh():
    params = build_params("thresh", CONFIG, seed_index=0, n_input=100, device=DEVICE)
    o_E = torch.full((CONFIG.n_excitatory, 4), 3.0)
    new = update(params, o_E, None, targ=1.4, eta=1e-1)
    assert torch.all(new.theta > params.theta)


def test_update_is_pure():
    params = build_params("rec", CONFIG, seed_index=0, n_input=100, device=DEVICE)
    before = params.w_EI.clone()
    update(params, torch.ones((CONFIG.n_excitatory, 4)), torch.ones((1, 4)), targ=0.2, eta=1.0)
    assert torch.equal(params.w_EI, before)


def test_eta_schedule_matches_original():
    assert eta_schedule(1.0, 0, 0.95, 3) == 1.0
    assert eta_schedule(1.0, 3, 0.95, 3) == 1.0
    assert abs(eta_schedule(1.0, 4, 0.95, 3) - 0.95) < 1e-12
    assert abs(eta_schedule(1.0, 9, 0.95, 3) - 0.95**6) < 1e-12


def test_exact_zero_fraction_is_meaningful():
    """ReLU output must contain exact zeros (basis of zero-fraction metrics)."""
    params = build_params("rec", CONFIG, seed_index=0, n_input=100, device=DEVICE)
    o_E = settle(params, _batch(), CONFIG).o_E
    zero_frac = float((o_E == 0.0).float().mean())
    assert 0.0 < zero_frac < 1.0
