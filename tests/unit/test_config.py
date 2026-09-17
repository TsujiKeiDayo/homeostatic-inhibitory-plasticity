import pytest
from hnn2.config import ExperimentConfig, RunSpec, config_to_json, experiment_config_from_json

def test_json_round_trip_preserves_equality():
    config = ExperimentConfig(n_epochs=3, image_size=(8, 8))
    restored = experiment_config_from_json(config_to_json(config))
    assert restored == config


def test_run_spec_rejects_unknown_model():
    with pytest.raises(ValueError):
        RunSpec(model_code="mlp", targ=0.2, eta=1e-3, seed_index=0)


def test_low_targ_warns_but_does_not_raise():
    """V7: the old hard constraint targ > theta_init is deliberately absent."""
    config = ExperimentConfig()
    spec = RunSpec(model_code="rec", targ=0.1, eta=1e-3, seed_index=0)
    with pytest.warns(UserWarning, match="theta_init"):
        spec.validate_against(config)


def test_original_regime_boundary_is_theta_init():
    """D24 (2026-09-05): the acceptance gates A5 / A7 read only runs with
    targ > theta_init, the same boundary V7 warns about."""
    config = ExperimentConfig()
    assert not RunSpec("rec", 0.13, 1e-3, 0).in_original_regime(config)
    assert not RunSpec("rec", config.theta_init, 1e-3, 0).in_original_regime(config)
    assert RunSpec("rec", 0.2, 1e-3, 0).in_original_regime(config)
    with pytest.warns(UserWarning, match="theta_init"):
        RunSpec("ff", 0.05, 1e-4, 0).validate_against(config)


def test_invalid_split_names_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(monitor_probe_split="probe")


