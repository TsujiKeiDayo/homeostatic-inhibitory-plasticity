"""T2: the settling loop must reach the analytic fixed point (design §7).

Internal consistency of the NEW implementation — not a comparison against the
old code. ff/thresh have closed forms; rec's fixed point is conditioned on the
settled ReLU support.
"""

import numpy as np
import pytest
import torch

from hnn2.config import ExperimentConfig
from hnn2.model.dynamics import Decays, analytic_fixed_point
from hnn2.model.init import build_params
from hnn2.simulate import settle

CONFIG = ExperimentConfig()
DEVICE = torch.device("cpu")


def _batch(n_input=100, batch=8, seed=123):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(
        rng.random((n_input, batch)), dtype=torch.float32, device=DEVICE
    )


@pytest.mark.parametrize("model_code", ["thresh", "ff", "rec"])
def test_settled_state_matches_analytic_fixed_point(model_code):
    params = build_params(model_code, CONFIG, seed_index=0, n_input=100, device=DEVICE)
    x = _batch()
    state = settle(params, x, CONFIG)

    decays = Decays.from_taus(CONFIG.tau_e, CONFIG.tau_i)
    fp = analytic_fixed_point(params, x, decays, o_E_settled=state.o_E)

    rel_err = (state.v_E - fp["v_E"]).abs().max() / state.v_E.abs().max()
    assert float(rel_err) < 1e-3, f"{model_code}: v_E rel err {float(rel_err):.2e}"

    if "v_I" in fp:
        rel_err_i = (state.v_I - fp["v_I"]).abs().max() / (state.v_I.abs().max() + 1e-12)
        assert float(rel_err_i) < 1e-3, f"{model_code}: v_I rel err {float(rel_err_i):.2e}"


def test_settling_is_deterministic():
    params = build_params("rec", CONFIG, seed_index=0, n_input=100, device=DEVICE)
    x = _batch()
    a = settle(params, x, CONFIG)
    b = settle(params, x, CONFIG)
    assert torch.equal(a.o_E, b.o_E)


def test_step_count_is_exactly_settle_steps():
    """V1: settle_steps counts state updates. One update from zeros must equal
    the map applied once by hand."""
    config = ExperimentConfig(settle_steps=1)
    params = build_params("thresh", config, seed_index=0, n_input=100, device=DEVICE)
    x = _batch()
    state = settle(params, x, config)
    expected_v_E = params.w_EIn @ x  # dec_E * 0 + drive
    assert torch.allclose(state.v_E, expected_v_E, atol=1e-6)
