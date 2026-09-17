"""HP handoff: full target coverage, shared entrypoints, and saved-result reuse."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from hnn2.config import ExperimentConfig, RunSpec
from hnn2.hp.rules import SelectionRule
from hnn2.result_io import read_single
from hnn2.workflows import read_selection, run_sweep_selection, run_selected_experiments, select_sweep

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "tests/fixtures/m3_reference/dataset.npz"
CONFIG = ExperimentConfig(n_excitatory=12, settle_steps=35, n_stabilise=2, batch_size=7,
                          n_epochs=3, readout_epochs=3, readout_eval_every=2)
SWEEP_CONFIG = replace(CONFIG, n_epochs=5)
RULE = SelectionRule(at_epoch=2, stability_epochs=2)
TARGETS = (.55, .60, .65, .70, .75)
SPECS = [RunSpec(model, targ, eta, seed) for model in ("rec", "ff", "thresh")
         for targ in TARGETS for eta in (.001, .003) for seed in (0, 1)]


def entry(name, **overrides):
    main = runpy.run_path(str(ROOT / "scripts" / f"{name}.py"))["main"]
    main.__globals__.update(overrides)
    main()


@pytest.fixture(scope="module")
def handoff(tmp_path_factory):
    root = tmp_path_factory.mktemp("hp_handoff")
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        separate = root / "separate"
        entry("run_sweep", DATASET=DATASET, CONFIG=SWEEP_CONFIG, SPECS=SPECS, RULE=RULE,
              TRAINING_CONFIG=CONFIG, OUTPUT=separate / "sweep", SELECTION_OUTPUT=separate / "selection")
        entry("run_experiments", DATASET=DATASET, CONFIG=CONFIG,
              SOURCE=separate / "selection", OUTPUT=separate / "trained")
        together = root / "together"
        entry("run_pipeline", DATASET=DATASET, CONFIG=CONFIG, SWEEP_CONFIG=SWEEP_CONFIG,
              SWEEP_SPECS=SPECS, RULE=RULE, SWEEP_OUTPUT=together / "sweep",
              SELECTION_OUTPUT=together / "selection", EXPERIMENTS_OUTPUT=together / "trained",
              BASELINES_OUTPUT=together / "baselines")
        yield root
    finally:
        torch.set_num_threads(original_threads)


def test_all_five_targets_use_selected_eta_and_hp_seeds(handoff):
    root = handoff / "separate"
    selection, _ = read_selection(root / "selection")
    states = pd.read_csv(root / "trained/last_run.csv", float_precision="round_trip")
    expected = {(model, targ, seed) for model in ("rec", "ff", "thresh")
                for targ in TARGETS for seed in (0, 1)}
    assert set(states[["model_code", "targ", "seed_index"]].itertuples(index=False, name=None)) == expected
    assert len(states) == 30 and states.state.eq("complete").all()
    assert selection["schedule"]["n_epochs"] == 3
    assert selection["rule"]["at_epoch"] == 2
    assert selection["sweep_config"]["n_epochs"] == 5
    digest = hashlib.sha256((root / "selection/selection.json").read_bytes()).hexdigest()
    for row in states.itertuples():
        cell = next(c for c in selection["cells"].values()
                    if c["model_code"] == row.model_code and c["targ"] == row.targ)
        conditions, arrays, tables = read_single(row.path)
        assert row.eta == cell["selected_eta"] == conditions["selection"]["used_eta"]
        assert conditions["selection"]["selected_eta"] == row.eta
        assert conditions["selection"]["source"]["selection_sha256"] == digest
        assert conditions["config"]["n_epochs"] == 3
        assert len(tables["monitor"]) == 3
        assert {"h_train", "h_val", "h_test", "h_test_full", "y_test_full"} <= arrays["features"].keys()


def test_separate_and_combined_entries_match_numerically(handoff):
    left = pd.read_csv(handoff / "separate/trained/last_run.csv")
    right = pd.read_csv(handoff / "together/trained/last_run.csv")
    pd.testing.assert_frame_equal(left.drop(columns="path"), right.drop(columns="path"))
    for one, two in zip(left.path, right.path):
        _, arrays_a, tables_a = read_single(one)
        _, arrays_b, tables_b = read_single(two)
        for name, arrays in arrays_a.items():
            for key, value in arrays.items():
                np.testing.assert_array_equal(value, arrays_b[name][key])
        for name, table in tables_a.items():
            pd.testing.assert_frame_equal(table, tables_b[name], check_exact=True)


def test_pipeline_reuses_completed_hp_and_training(handoff, monkeypatch):
    import hnn2.workflows as workflows
    root = handoff / "together"
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file() and p.name != 'last_run.csv'}
    def forbidden(*args, **kwargs):
        pytest.fail("Completed pipeline tried to execute HP or adaptation again")
    monkeypatch.setattr(workflows, "run_sweep", forbidden)
    monkeypatch.setattr(workflows, "run_adaptation_batch", forbidden)
    entry("run_pipeline", DATASET=DATASET, CONFIG=CONFIG, SWEEP_CONFIG=SWEEP_CONFIG,
          SWEEP_SPECS=SPECS, RULE=RULE, SWEEP_OUTPUT=root / "sweep",
          SELECTION_OUTPUT=root / "selection", EXPERIMENTS_OUTPUT=root / "trained",
          BASELINES_OUTPUT=root / "baselines")
    assert pd.read_csv(root / "trained/last_run.csv").state.eq("skipped").all()
    assert all(p.read_bytes() == value for p, value in before.items())


def test_training_only_never_calls_hp(handoff, tmp_path, monkeypatch):
    import hnn2.workflows as workflows
    calls = []
    def capture(dataset, specs, config, directory, **kwargs):
        calls.extend(specs)
        Path(directory).mkdir()
        return pd.DataFrame([{"state": "complete"} for _ in specs])
    def forbidden(*args, **kwargs):
        pytest.fail("Training-only tried to call HP")
    monkeypatch.setattr(workflows, "run_sweep", forbidden)
    monkeypatch.setattr(workflows, "run_experiments", capture)
    run_selected_experiments(handoff / "separate/selection", DATASET, CONFIG, tmp_path / "trained")
    assert len(calls) == 30
    assert {s.targ for s in calls} == set(TARGETS)


def test_reselection_preserves_explicit_training_length(handoff, tmp_path, monkeypatch):
    """An earlier HP scoring point must not shorten training (no new network run)."""
    import hnn2.workflows as workflows
    source = handoff / "separate/selection"
    curves = pd.read_csv(source / "curves.csv", float_precision="round_trip")
    saved = json.loads((source / "conditions.json").read_text(encoding="utf-8"))
    select_sweep(curves, SPECS, SWEEP_CONFIG, replace(RULE, at_epoch=1), tmp_path / "selection",
                 training_epochs=3, input_source=saved["input"])
    selection, _ = read_selection(tmp_path / "selection")
    assert selection["rule"]["at_epoch"] == 1
    assert selection["schedule"]["n_epochs"] == 3
    def capture(dataset, specs, config, directory, **kwargs):
        assert config.n_epochs == 3
        Path(directory).mkdir()
        return pd.DataFrame([{"state": "complete"} for _ in specs])
    monkeypatch.setattr(workflows, "run_experiments", capture)
    run_selected_experiments(tmp_path / "selection", DATASET, CONFIG, tmp_path / "trained")


def test_incomplete_selection_restarts_from_saved_candidates(handoff, tmp_path, monkeypatch):
    import hnn2.workflows as workflows
    source = handoff / "separate"
    target = tmp_path / "selection"
    shutil.copytree(source / "selection", target)
    (target / "COMPLETE").unlink()
    before = (target / "selection.json").read_bytes()
    monkeypatch.setattr(workflows, "run_adaptation_batch", lambda *a, **k: pytest.fail("Completed HP candidate reran"))
    run_sweep_selection(DATASET, SPECS, SWEEP_CONFIG, RULE, source / "sweep", target, training_config=CONFIG,
                        batch_runs=2)
    assert (target / "COMPLETE").is_file()
    assert (target / "selection.json").read_bytes() == before


@pytest.mark.parametrize("damage", ["missing_cell", "duplicate_cell", "wrong_eta", "missing_file", "incomplete"])
def test_damaged_selection_stops_before_training(handoff, tmp_path, monkeypatch, damage):
    import hnn2.workflows as workflows
    path = tmp_path / "selection"
    shutil.copytree(handoff / "separate/selection", path)
    if damage in ("missing_file", "incomplete"):
        (path / ("selected.csv" if damage == "missing_file" else "COMPLETE")).unlink()
    else:
        file = path / "selection.json"
        saved = json.loads(file.read_text(encoding="utf-8"))
        key = next(iter(saved["cells"]))
        if damage == "missing_cell":
            del saved["cells"][key]
        elif damage == "duplicate_cell":
            saved["cells"]["duplicate"] = saved["cells"][key]
        else:
            saved["cells"][key]["selected_eta"] = .02
        file.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(workflows, "run_experiments", lambda *a, **k: pytest.fail("Invalid selection reached training"))
    with pytest.raises(ValueError):
        run_selected_experiments(path, DATASET, CONFIG, tmp_path / "trained")
    assert not (tmp_path / "trained").exists()


@pytest.mark.parametrize("changes", [{"tau_i": 21}, {"batch_size": 8}, {"n_epochs": 4}, {"eta_decay": .9}])
def test_changed_training_conditions_stop_before_computation(handoff, tmp_path, monkeypatch, changes):
    import hnn2.workflows as workflows
    monkeypatch.setattr(workflows, "run_experiments", lambda *a, **k: pytest.fail("Invalid config reached training"))
    with pytest.raises(ValueError):
        run_selected_experiments(handoff / "separate/selection", DATASET, replace(CONFIG, **changes), tmp_path / "trained")


def test_changed_input_and_changed_hp_rule_are_rejected(handoff, tmp_path, monkeypatch):
    import hnn2.workflows as workflows
    from hnn2.data import load_dataset, Dataset
    data = load_dataset(DATASET)
    changed = Dataset(data.arrays, {**data.source, "sha256": "different"})
    monkeypatch.setattr(workflows, "run_adaptation_batch", lambda *a, **k: pytest.fail("Different input reached learning"))
    with pytest.raises(ValueError, match="different input"):
        run_selected_experiments(handoff / "separate/selection", changed, CONFIG, tmp_path / "trained")
    monkeypatch.setattr(workflows, "run_sweep", lambda *a, **k: pytest.fail("Different rule reached HP"))
    with pytest.raises(ValueError, match="Conditions differ"):
        run_sweep_selection(DATASET, SPECS, SWEEP_CONFIG, replace(RULE, stability_tol=.3),
                            tmp_path / "sweep", handoff / "separate/selection", training_config=CONFIG)
    with pytest.raises(ValueError, match="Conditions differ"):
        run_sweep_selection(DATASET, SPECS, SWEEP_CONFIG, RULE, tmp_path / "sweep",
                            handoff / "separate/selection", training_config=replace(CONFIG, n_epochs=4))


def test_training_failure_is_not_reported_as_pipeline_success(handoff, tmp_path, monkeypatch):
    import hnn2.workflows as workflows
    def failed(dataset, specs, config, directory, **kwargs):
        Path(directory).mkdir()
        return pd.DataFrame([{"state": "failed", "reason": "test failure"}])
    monkeypatch.setattr(workflows, "run_experiments", failed)
    with pytest.raises(RuntimeError, match="Some conditions failed"):
        run_selected_experiments(handoff / "separate/selection", DATASET, CONFIG, tmp_path / "trained")
    assert pd.read_csv(tmp_path / "trained/last_run.csv").state.eq("failed").all()
