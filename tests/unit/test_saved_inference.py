"""Saved trajectories and local equations, without any learning update."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_fixtures import fixed_result
from hnn2.data import load_dataset
from hnn2.result_io import save_single, read_single, SaveOptions
from hnn2.saved_inference import training_order_activity, fixed_point_detail, local_modes
from hnn2.model.params import ModelParams, State
from hnn2.model.dynamics import Decays, input_drive, step


@pytest.fixture(autouse=True)
def no_learning(monkeypatch):
    import hnn2.adapt as adapt
    import hnn2.model.plasticity as plasticity
    import hnn2.single as single
    def forbidden(*args, **kwargs):
        pytest.fail("Saved inference must never perform training")
    monkeypatch.setattr(adapt, "update", forbidden)
    monkeypatch.setattr(plasticity, "update", forbidden)
    monkeypatch.setattr(single, "fit_readout", forbidden)


@pytest.mark.parametrize("model", ["rec", "ff", "thresh"])
def test_saved_replay_and_probe_identity(tmp_path, model):
    result = fixed_result(model)
    save_single(result, tmp_path / model, saving=SaveOptions(save_full_test=True))
    conditions, arrays, _ = read_single(tmp_path / model)
    dataset = load_dataset(conditions["input"]["path"])
    activity, checks = training_order_activity(conditions, arrays, dataset)
    assert checks.equal.all(), checks.to_dict("records")
    n = len(dataset.arrays["y_train"])
    assert len(activity) == n * conditions["config"]["n_epochs"]
    for _, epoch in activity.groupby("epoch"):
        assert sorted(epoch.sample_index) == list(range(n))
    assert np.array_equal(activity.order_index, np.arange(len(activity)))
    detail, check, states = fixed_point_detail(conditions, arrays, dataset)
    assert check.equal.all(), check.to_dict("records")
    assert len(detail) == len(dataset.arrays["y_test"])
    assert set(states) == {"v_E", "v_I", "o_E"}
    if model == "thresh":
        assert detail.I_reason.eq("not_applicable").all()
        assert detail.I_relative_error.isna().all()


@pytest.mark.parametrize("model", ["rec", "ff", "thresh"])
def test_local_eigenvalues_match_the_actual_map_jacobian(model):
    dtype = torch.float64
    params = ModelParams(model, torch.tensor([[.2], [.3], [.4]], dtype=dtype),
                         torch.tensor([.1, .1, .1], dtype=dtype))
    if model != "thresh":
        params = params.with_(w_EI=torch.tensor([[.1], [.2], [.3]], dtype=dtype))
        params = params.with_(**({"w_IE": torch.tensor([[.2, .3, .1]], dtype=dtype)} if model == "rec"
                                else {"w_IIn": torch.tensor([[.2]], dtype=dtype)}))
    decays = Decays.from_taus(2, 20)
    x = torch.ones((1, 1), dtype=dtype)
    point = torch.tensor([.3, -.2, .5] + ([.2] if model != "thresh" else []), dtype=dtype)
    def transition(values):
        state = State(values[:3, None], (values[:3, None]-params.theta[:, None]).clamp(min=0), values[3:, None])
        updated = step(params, state, input_drive(params, x), decays, x)
        return torch.cat([updated.v_E[:, 0], updated.v_I[:, 0]])
    jacobian = torch.autograd.functional.jacobian(transition, point).numpy()
    state = State(point[:3, None], (point[:3, None]-params.theta[:, None]).clamp(min=0), point[3:, None])
    row = local_modes(params, state, decays).iloc[0]
    expected_modes = [complex(row.lambda1_real, row.lambda1_imag)]
    if model != "thresh":
        expected_modes.append(complex(row.lambda2_real, row.lambda2_imag))
    copies = params.n_excitatory - 1
    expected_modes.extend([decays.dec_E] * copies)
    np.testing.assert_allclose(np.sort_complex(expected_modes), np.sort_complex(np.linalg.eigvals(jacobian)), atol=1e-12)
    assert row.spectral_radius == pytest.approx(max(abs(np.linalg.eigvals(jacobian))))
    assert row.mode_decay_steps == pytest.approx(np.log(1e-4)/np.log(row.spectral_radius))
    assert row.active_units == 2


def test_relu_boundary_and_noncontracting_modes_are_undefined():
    params = ModelParams("rec", torch.ones((2, 1)), torch.zeros(2),
                         w_EI=torch.ones((2, 1))*3, w_IE=torch.ones((1, 2)))
    state = State(torch.tensor([[0., 1.], [1., 1.]]), torch.tensor([[0., 1.], [1., 1.]]), torch.ones((1, 2)))
    detail = local_modes(params, state, Decays.from_taus(2, 20))
    assert detail.undefined_reason.tolist() == ["relu_boundary", "noncontracting"]
    assert detail.mode_decay_steps.isna().all()


def test_replay_restores_preupdate_parameters_epoch_warm_state_and_shuffle(tmp_path, monkeypatch):
    import hnn2.saved_inference as inference
    from hnn2.rng import stream
    result = fixed_result("ff")
    save_single(result, tmp_path / "ff", saving=SaveOptions(save_full_test=True))
    conditions, arrays, _ = read_single(tmp_path / "ff")
    trajectory = arrays["summaries"]["param_trajectory"]
    # Distinct snapshots expose one-step offsets hidden by a constant fixture.
    trajectory[:] += np.arange(len(trajectory))[:, None] * 1e-4
    dataset = load_dataset(conditions["input"]["path"])
    calls, warms = [], []
    encode, warm_start = inference.encode, inference.warm_start_state
    def tracked_encode(params, x, config, warm):
        calls.append((params.w_EI.cpu().numpy().ravel().copy(), x.cpu().numpy().copy(), warm))
        return encode(params, x, config, warm)
    def tracked_warm(params, train, config):
        warm = warm_start(params, train, config)
        warms.append((params.w_EI.cpu().numpy().ravel().copy(), warm))
        return warm
    monkeypatch.setattr(inference, "encode", tracked_encode)
    monkeypatch.setattr(inference, "warm_start_state", tracked_warm)
    activity, _ = inference.training_order_activity(conditions, arrays, dataset)
    config = conditions["config"]
    n, batch = len(dataset.arrays["y_train"]), config["batch_size"]
    rng = stream(conditions["spec"]["seed_index"], "shuffle")
    call_index, warm_index, parameter_index = 0, 0, 0
    for epoch in range(config["n_epochs"]):
        order = rng.permutation(n)
        np.testing.assert_array_equal(activity.loc[activity.epoch.eq(epoch), "sample_index"], order)
        np.testing.assert_array_equal(warms[warm_index][0], trajectory[parameter_index])
        for start in range(0, n, batch):
            weights, x, warm = calls[call_index]
            np.testing.assert_array_equal(weights, trajectory[parameter_index])
            np.testing.assert_array_equal(x, dataset.arrays["x_train_small"][order[start:start+batch]].T)
            assert warm is warms[warm_index][1]
            call_index += 1
            parameter_index += 1
        warm_index += 1
        if epoch in (0, config["n_epochs"]-1):
            np.testing.assert_array_equal(calls[call_index][0], trajectory[parameter_index])
            call_index += 1
            warm_index += 1
    assert call_index == len(calls) and warm_index == len(warms)


def test_fixed_point_zero_denominators_remain_undefined(tmp_path, monkeypatch):
    import hnn2.saved_inference as inference
    result = fixed_result("ff")
    save_single(result, tmp_path / "ff", saving=SaveOptions(save_full_test=True))
    conditions, arrays, _ = read_single(tmp_path / "ff")
    dataset = load_dataset(conditions["input"]["path"])
    def zero_state(params, x, config, warm):
        return State(torch.zeros((params.n_excitatory, x.shape[1])),
                     torch.zeros((params.n_excitatory, x.shape[1])),
                     torch.zeros((params.n_inhibitory, x.shape[1])))
    monkeypatch.setattr(inference, "encode", zero_state)
    detail, _, _ = inference.fixed_point_detail(conditions, arrays, dataset)
    for prefix in ("E", "I"):
        assert detail[f"{prefix}_reason"].eq("zero_denominator").all()
        assert detail[f"{prefix}_relative_error"].isna().all()


def test_repeated_and_real_roots_are_distinguished():
    params = ModelParams("rec", torch.ones((2, 1)), torch.zeros(2),
                         w_EI=torch.zeros((2, 1)), w_IE=torch.ones((1, 2)))
    state = State(torch.ones((2, 1)), torch.ones((2, 1)), torch.ones((1, 1)))
    repeated = local_modes(params, state, Decays.from_taus(2, 2)).iloc[0]
    assert repeated.root_kind == "repeated_root"
    assert repeated.undefined_reason == "repeated_root" and np.isnan(repeated.mode_decay_steps)
    real = local_modes(params, state, Decays.from_taus(2, 20)).iloc[0]
    assert real.root_kind == "real_roots" and real.undefined_reason == ""
