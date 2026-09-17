"""Control flow and persistence only: no HP, plasticity or classifier training."""
from dataclasses import replace
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from hnn2.config import ExperimentConfig, RunSpec
from hnn2.readout import ReadoutHistory, ReadoutResult
from hnn2.result_io import SaveOptions, read_single
from hnn2.single import MlpSettings, SingleResult, _baseline_conditions
import hnn2.workflows as workflows

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from figure_fixtures import fixed_result
BASE = ROOT / "tests/fixtures/m3_reference"
FROZEN = json.loads((BASE / "conditions.json").read_text(encoding="utf-8"))
CONFIG = ExperimentConfig(**{k: tuple(v) if k in ("image_size", "raw_baseline_size") else v
                            for k, v in FROZEN["config"].items()})
MLP = MlpSettings(**FROZEN["mlp"])


@pytest.fixture(autouse=True)
def forbid_learning(monkeypatch):
    import hnn2.single as single
    import hnn2.mlp as mlp
    original_threads = torch.get_num_threads()
    def forbidden(*args, **kwargs):
        pytest.fail("This test must not execute scientific training")
    monkeypatch.setattr(workflows, "run_adaptation_batch", forbidden)
    monkeypatch.setattr(workflows, "run_representation", forbidden)
    monkeypatch.setattr(single, "run_adaptation", forbidden)
    monkeypatch.setattr(single, "fit_readout", forbidden)
    monkeypatch.setattr(mlp, "train_mlp_e2e", forbidden)
    yield
    torch.set_num_threads(original_threads)


def entry(name):
    return runpy.run_path(str(ROOT / "scripts" / f"{name}.py"))["main"]


@pytest.mark.parametrize("failure", [None, "hp", "training", "baselines"])
def test_pipeline_order_and_failure_stops(failure, monkeypatch, tmp_path):
    main = entry("run_pipeline")
    namespace = main.__globals__
    dataset, calls = object(), []
    selected = pd.DataFrame([
        {"model_code": model, "targ": targ, "eta": .001, "seed_index": seed,
         "state": "complete", "reason": ""}
        for model in ("ff", "rec") for targ in (.55, .65) for seed in (1, 4)])
    def step(name, result):
        def call(*args, **kwargs):
            calls.append(name)
            assert (args[1] if name == "training" else args[0]) is dataset
            assert kwargs["settings_path"] == namespace["SETTINGS_PATH"]
            if name == "baselines":
                assert args[1:3] == (["ff", "rec"], [1, 4])
                assert args[3] is namespace["CONFIG"]
                assert kwargs["saving"] is namespace["SAVING"]
                assert kwargs["mlp"] is namespace["MLP"]
                assert kwargs["representations"] == namespace["BASELINE_REPRESENTATIONS"]
            if failure == name:
                raise RuntimeError(name)
            return result
        return call
    monkeypatch.setitem(namespace, "load_dataset", lambda path: dataset)
    monkeypatch.setitem(namespace, "SWEEP_SPECS", [RunSpec(r.model_code, r.targ, r.eta, r.seed_index)
                                                 for r in selected.itertuples()])
    monkeypatch.setitem(namespace, "run_sweep_selection", step("hp", {}))
    monkeypatch.setitem(namespace, "run_selected_experiments", step("training", selected))
    monkeypatch.setitem(namespace, "run_baselines", step("baselines", pd.DataFrame(
        [{"representation": "raw", "seed_index": 1, "state": "complete", "reason": ""}])))
    for key in ("SWEEP_OUTPUT", "SELECTION_OUTPUT", "EXPERIMENTS_OUTPUT", "BASELINES_OUTPUT"):
        monkeypatch.setitem(namespace, key, tmp_path / key)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            main()
    else:
        main()
    expected = ["hp", "training", "baselines"]
    if failure:
        expected = expected[:expected.index(failure) + 1]
    assert calls == expected
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) == 1 and files[0].name == "expected_runs.csv"
    planned = pd.read_csv(files[0])
    assert len(planned) == 8 and set(planned.seed_index) == {1, 4}


def frozen_result(dataset, config, seed, *, representation, model_code, mlp, analysis, device):
    """Checked-in arrays and hand-authored final evaluations, without training."""
    result = fixed_result(model_code, seed, representation=representation)
    conditions = _baseline_conditions(dataset, RunSpec(model_code, 1., 1., seed), config,
                                      analysis, device, representation, mlp)
    result.conditions = conditions
    return result


@pytest.mark.parametrize("saving", [SaveOptions(save_full_test=True), SaveOptions(
    save_features=False, save_encoder_weights=False, save_activity=False,
    save_summaries=False, save_readout_history=False)])
def test_baseline_coverage_saving_and_reuse(saving, monkeypatch, tmp_path):
    calls = []
    def produce(*args, **kwargs):
        calls.append((kwargs["representation"], kwargs["model_code"], args[2]))
        return frozen_result(*args, **kwargs)
    monkeypatch.setattr(workflows, "run_representation", produce)
    def run(**kwargs):
        return workflows.run_baselines(BASE / "dataset.npz", ("rec", "ff", "thresh"), (0, 1),
                                       CONFIG, tmp_path / "baselines", mlp=MLP, saving=saving, **kwargs)
    states = run()
    assert len(states) == 10 and states.state.eq("complete").all()
    assert len(calls) == len(set(calls)) == 10
    for row in states.itertuples():
        conditions, arrays, tables = read_single(row.path)
        assert "targ" not in conditions["spec"] and "eta" not in conditions["spec"]
        assert "summary" in tables
        assert ("features" in arrays) == saving.save_features
        assert conditions["readout_streams"]["init"] == ("mlp/readout_init" if row.representation == "mlp_frozen" else "readout/init")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and p.name != "last_run.csv"}
    assert run().state.eq("skipped").all()
    assert len(calls) == 10
    assert all(p.read_bytes() == content for p, content in before.items())
    with pytest.raises(ValueError, match="Conditions differ|population differs"):
        workflows.run_baselines(BASE / "dataset.npz", ("rec",), (0,), replace(CONFIG, readout_lr=.123),
                                tmp_path / "baselines", mlp=MLP, saving=saving)
    (Path(states.path.iloc[0]) / "summary.csv").unlink()
    with pytest.raises(ValueError, match="Completed result lacks|Incomplete single result"):
        run()
    assert len(calls) == 10


def test_baseline_failure_is_recorded_and_only_failed_condition_retries(monkeypatch, tmp_path):
    calls = []
    fail = True
    def produce(*args, **kwargs):
        calls.append(kwargs["representation"])
        if fail and kwargs["representation"] == "mlp_frozen":
            raise RuntimeError("simulated classifier failure")
        return frozen_result(*args, **kwargs)
    monkeypatch.setattr(workflows, "run_representation", produce)
    def run():
        return workflows.run_baselines(BASE / "dataset.npz", ("rec",), (0,), CONFIG, tmp_path, mlp=MLP)
    with pytest.raises(RuntimeError, match="Some baselines failed"):
        run()
    assert pd.read_csv(tmp_path / "last_run.csv").state.tolist() == ["complete", "complete", "failed"]
    assert (tmp_path / "mlp_frozen/seed-0/FAILED.json").is_file()
    assert not (tmp_path / "mlp_frozen/seed-0/COMPLETE").exists()
    fail = False
    assert run().state.tolist() == ["skipped", "skipped", "complete"]
    assert calls == ["sparse_initial", "raw", "mlp_frozen", "mlp_frozen"]
    assert not (tmp_path / "mlp_frozen/seed-0/FAILED.json").exists()


@pytest.mark.parametrize("models,seeds,reps", [(("bad",), (0,), ()), (("rec",), (0, 0), ()),
                                             (("rec",), (True,), ()), (("rec",), (0,), ("bad",))])
def test_invalid_baseline_coordinates_stop_before_io(models, seeds, reps, tmp_path):
    with pytest.raises(ValueError):
        workflows.run_baselines("missing.npz", models, seeds, CONFIG, tmp_path / "out", representations=reps)
    assert not (tmp_path / "out").exists()


def test_standalone_baseline_uses_shared_function(monkeypatch, tmp_path):
    main = entry("run_baselines")
    calls = []
    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return pd.DataFrame(columns=["representation", "seed_index", "state", "reason"])
    monkeypatch.setitem(main.__globals__, "run_baselines", capture)
    monkeypatch.setitem(main.__globals__, "OUTPUT", tmp_path)
    main()
    args, kwargs = calls[0]
    assert args[1:3] == (main.__globals__["MODELS"], main.__globals__["SEEDS"])
    assert kwargs["saving"] is main.__globals__["SAVING"]
    assert not list(tmp_path.iterdir())
