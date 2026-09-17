"""Only fixed arrays and stubbed computation. No real learning or inference."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notebook_cells import threshold_results
from figure_fixtures import fixed_result, CONFIG, BASE
from hnn2.artifact_checks import inspect_saved, output_issues, readout_issues
from hnn2.data import load_dataset
from hnn2.result_io import SaveOptions, save_single, read_single, completed, single_files, write_result


@pytest.fixture(autouse=True)
def no_learning(monkeypatch):
    import hnn2.single as single
    import hnn2.workflows as workflows
    import hnn2.mlp as mlp
    import hnn2.postprocess as postprocess
    import hnn2.encoders as encoders
    def forbidden(*a, **k):
        pytest.fail("Actual learning/inference forbidden")
    for module, name in ((single, "run_adaptation"), (single, "fit_readout"),
                         (single, "sparse_features"), (single, "run_representation"),
                         (workflows, "run_adaptation_batch"), (workflows, "finish_adaptation"),
                         (workflows, "run_representation"), (mlp, "train_mlp_e2e"),
                         (postprocess, "fit_readout"), (encoders, "sparse_features"),
                         (encoders, "mlp_frozen_features"), (encoders, "raw_features")):
        monkeypatch.setattr(module, name, forbidden)


@pytest.mark.parametrize("model", ["rec", "ff", "thresh"])
def test_fixed_current_roundtrip_and_trajectory(tmp_path, model):
    r = fixed_result(model)
    path = save_single(r, tmp_path / model, saving=SaveOptions(save_full_test=True))
    assert inspect_saved(path, dataset=load_dataset(BASE / "dataset.npz")) == []
    c, a, t = read_single(path)
    assert t["summary"].final_val_loss.iloc[0] == r.readout.final_val_loss
    assert t["summary"].final_step.iloc[0] == 9
    assert t["history"].val_evaluated.tolist() == [False, True, False, True, False, True, False, True, False]
    assert t["summary"].final_val_accuracy.iloc[0] != t["history"].val_accuracy.iloc[-1]
    assert c["numerics"]["n_I"] == (0 if model == "thresh" else 1)
    assert "src/hnn2/readout.py" in c["environment"]["source_sha256"]
    for split in ("train", "val", "test", "test_full"):
        np.testing.assert_array_equal(a["features"][f"h_{split}"], r.features[f"h_{split}"])
        np.testing.assert_array_equal(a["features"][f"sample_index_{split}"], np.arange(len(r.features[f"h_{split}"])))


@pytest.mark.parametrize("off", ["save_features", "save_full_test", "save_encoder_weights", "save_activity", "save_summaries", "save_readout_history", "all"])
def test_switches_preserve_values_and_explicit_shortages(tmp_path, off):
    r = fixed_result()
    all_on = SaveOptions(save_full_test=True)
    flags = all_on.__dict__.copy()
    if off == "all":
        flags = dict.fromkeys(flags, False)
    else:
        flags[off] = False
    full = save_single(r, tmp_path / "full", saving=all_on)
    path = save_single(r, tmp_path / "off", saving=SaveOptions(**flags))
    c, a, t = read_single(path)
    assert output_issues(c, a, t) == []
    assert inspect_saved(path)
    _, aa, tt = read_single(full)
    for name, values in a.items():
        for key, value in values.items():
            np.testing.assert_array_equal(value, aa[name][key])
    for name, value in t.items():
        import pandas as pd
        pd.testing.assert_frame_equal(value, tt[name])


@pytest.mark.parametrize("damage", ["missing_key", "dtype", "axis", "trajectory", "labels", "evaluation", "final_step",
                                    "monitor_epoch", "monitor_nonfinite", "entropy_bins"])
def test_damaged_current_values_are_reported(tmp_path, damage):
    path = save_single(fixed_result(), tmp_path / damage, saving=SaveOptions(save_full_test=True))
    c, a, t = read_single(path)
    if damage == "missing_key": del a["features"]["y_test"]
    if damage == "dtype": a["features"]["h_test"] = a["features"]["h_test"].astype("float64")
    if damage == "axis": a["features"]["h_train"] = a["features"]["h_train"].T
    if damage == "trajectory": a["summaries"]["param_trajectory"][0, 0] += 1
    if damage == "labels": a["features"]["y_test"][0] = 999
    if damage == "evaluation": t["history"].loc[0, "val_evaluated"] = True
    if damage == "final_step": t["summary"].loc[0, "final_step"] = 8
    if damage == "monitor_epoch": t["monitor"].loc[0, "epoch"] = 1
    if damage == "monitor_nonfinite": t["monitor"].loc[0, "composite"] = np.nan
    if damage == "entropy_bins": t["entropy"].loc[0, "width"] = 123.0
    assert output_issues(c, a, t, dataset=load_dataset(BASE / "dataset.npz"))


def test_missing_version_is_distinct_from_missing_options(tmp_path):
    # All switches and stream names are explicit; only certification is missing.
    path = save_single(fixed_result(), tmp_path / "legacy")
    c, _, _ = read_single(path)
    expected = dict(c)
    c.pop("output_version")
    (path / "conditions.json").write_text(json.dumps(c), encoding="utf-8")
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
    assert read_single(path)[0]["representation"] == "sparse_trained"
    assert "legacy" in inspect_saved(path)[0]
    with pytest.raises(ValueError, match="Legacy COMPLETE"):
        completed(path, expected, ["summary.csv"])
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}


@pytest.mark.parametrize("complete", [True, False])
@pytest.mark.parametrize("key", [*SaveOptions.__dataclass_fields__, "readout_streams"])
def test_missing_records_refused_without_retraining_or_repair(tmp_path, key, complete):
    from hnn2.config import RunSpec
    from hnn2.postprocess import saved_features, readout_only, features_from_encoder, compare_tables
    from hnn2.workflows import run_experiments, condition_directory
    spec = RunSpec("rec", .35, .001, 0)
    saving = SaveOptions(save_full_test=True)
    path = save_single(fixed_result(), condition_directory(tmp_path, spec), saving=saving)
    c, a, t = read_single(path)
    expected = dict(c)
    c.pop(key)
    (path / "conditions.json").write_text(json.dumps(c), encoding="utf-8")
    assert any(key in issue for issue in output_issues(c, a, t))
    assert any(key in issue for issue in readout_issues(c, t))
    with pytest.raises(ValueError, match=key):
        single_files(c)
    if complete:
        for call in (lambda: read_single(path), lambda: saved_features(path),
                     lambda: readout_only(path, CONFIG, tmp_path / "readout"),
                     lambda: features_from_encoder(path, BASE / "dataset.npz", CONFIG),
                     lambda: compare_tables([path], "summary")):
            with pytest.raises(ValueError, match=key):
                call()
    else:
        (path / "COMPLETE").unlink()
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
    with pytest.raises(ValueError, match=key):
        completed(path, expected, ["summary.csv"])
    with pytest.raises(ValueError, match=key):
        run_experiments(BASE / "dataset.npz", [spec], CONFIG, tmp_path, saving=saving)
    if not complete:
        with pytest.raises(ValueError, match=key):
            save_single(fixed_result(), path, saving=saving, resume=True)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
    assert not (tmp_path / "readout").exists()


@pytest.mark.parametrize("key", list(SaveOptions.__dataclass_fields__))
def test_recorded_switch_must_be_bool_not_integer(tmp_path, key):
    path = save_single(fixed_result(), tmp_path / "run")
    c, a, t = read_single(path)
    c[key] = 1  # Equal to True in Python, but not a recorded Boolean choice.
    (path / "conditions.json").write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(ValueError, match=key):
        read_single(path)
    assert any(key in issue for issue in output_issues(c, a, t))


@pytest.mark.parametrize("streams", [None, {}, {"init": "readout/init"},
                                    {"init": "", "shuffle": "readout/shuffle"}])
def test_partial_or_invalid_stream_record_is_not_completed(tmp_path, streams):
    path = save_single(fixed_result(), tmp_path / "run")
    c, _, _ = read_single(path)
    c["readout_streams"] = streams
    (path / "conditions.json").write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(ValueError, match="readout_streams"):
        read_single(path)


@pytest.mark.parametrize("representation", ["sparse_initial", "raw", "mlp_frozen"])
def test_baseline_records_are_required_too(tmp_path, representation):
    path = save_single(fixed_result(representation=representation), tmp_path / "run")
    c, _, _ = read_single(path)
    expected = dict(c)
    c.pop("save_features")
    (path / "conditions.json").write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(ValueError, match="save_features"):
        read_single(path)
    with pytest.raises(ValueError, match="save_features"):
        completed(path, expected, ["summary.csv"])


@pytest.mark.parametrize("representation", ["sparse_trained", "mlp_frozen"])
def test_readout_only_keeps_recorded_streams_and_its_own_contract(tmp_path, monkeypatch, representation):
    import hnn2.postprocess as postprocess
    result = fixed_result(representation=representation)
    source = save_single(result, tmp_path / "source", saving=SaveOptions(save_full_test=True))
    calls = []
    def fit(features, config, seed, **kwargs):
        calls.append(kwargs)
        return result.readout, {}  # No seed draws or classifier computation in this stub.
    monkeypatch.setattr(postprocess, "fit_readout", fit)
    output = tmp_path / "readout"
    postprocess.readout_only(source, CONFIG, output)
    c = json.loads((output / "conditions.json").read_text(encoding="utf-8"))
    assert c["readout_streams"] == result.conditions["readout_streams"]
    assert calls[0]["init_stream"] == c["readout_streams"]["init"]
    assert calls[0]["shuffle_stream"] == c["readout_streams"]["shuffle"]
    assert "save_features" not in c
    assert completed(output, c, ["summary.csv", "history.csv"])
    assert len(postprocess.compare_tables([output], "history")) == result.readout.final_step
    # Feature metrics have neither readout streams nor single saving switches.
    metrics = tmp_path / "metrics"
    postprocess.analyse_saved(source, metrics)
    m = json.loads((metrics / "conditions.json").read_text(encoding="utf-8"))
    assert "readout_streams" not in m and "save_features" not in m
    assert completed(metrics, m, ["entropy.csv", "silhouette.csv"])
    assert not postprocess.compare_tables([metrics], "entropy").empty


def test_readout_only_missing_history_flag_is_rejected(tmp_path):
    from hnn2.postprocess import compare_tables
    from hnn2.result_io import readout_tables
    r = fixed_result()
    c = {"representation": "readout_only", "input": r.conditions["input"],
         "config": asdict(CONFIG), "readout_streams": r.conditions["readout_streams"]}
    path = write_result(tmp_path / "readout", c, tables=readout_tables(r.readout))
    with pytest.raises(ValueError, match="save_readout_history"):
        compare_tables([path], "history")


@pytest.mark.parametrize("key", ["save_readout_history", "readout_streams"])
def test_rf05_loader_counts_missing_records_as_invalid(tmp_path, key):
    from figure_fixtures import fixed_experiment
    from hnn2.figure_tables import load_figure_runs
    root = fixed_experiment(tmp_path)
    path = root / "baselines/raw/seed-0/conditions.json"
    c = json.loads(path.read_text(encoding="utf-8"))
    c.pop(key)
    path.write_text(json.dumps(c), encoding="utf-8")
    runs, expected = load_figure_runs(root, readout_only=True)
    invalid = expected[expected.state == "invalid"]
    assert len(invalid) == 1 and key in invalid.reason.iloc[0]
    _, summary = threshold_results(runs, expected, dict.fromkeys(("rec", "ff", "thresh"), .35))
    raw = summary[summary.arm == "raw"]
    assert raw.expected_total.eq(2).all()
    assert raw.total.eq(1).all() and raw.missing_or_invalid.eq(1).all()


def test_incomplete_condition_mismatch_stops_before_work(tmp_path):
    c = fixed_result().conditions
    (tmp_path / "conditions.json").write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(ValueError, match="Conditions differ"):
        completed(tmp_path, {**c, "spec": {**c["spec"], "targ": .55}}, [])


def test_readout_returns_existing_final_evaluation_without_extra_calls(monkeypatch):
    """Zero loop iterations and stubbed evaluations: no optimizer step or inference."""
    import torch
    import hnn2.readout as readout
    calls = []
    scores = iter([(.234, .7), (.345, .6), (.456, .65)])
    def evaluate(*args):
        calls.append(1)
        return next(scores)
    monkeypatch.setattr(readout, "_evaluate", evaluate)
    # The operational config remains unchanged. Zero epochs is used solely to
    # exercise the existing final-evaluation return path without any learning.
    with torch.random.fork_rng(devices=[]):
        result = readout.train_readout(fixed_result().features, fixed_result().features,
            replace(CONFIG, readout_epochs=0), init_seed=1, shuffle_seed=2, device=torch.device("cpu"))
    assert len(calls) == 3
    assert result.final_val_loss == .234 and result.final_val_accuracy == .7
    assert result.final_step == 0 and result.history.step == []


def test_hp_handoff_and_save_with_all_learning_stubbed(tmp_path, monkeypatch):
    from figure_fixtures import fixed_experiment
    import hnn2.workflows as workflows
    source = fixed_experiment(tmp_path / "fixed")
    specs_seen = []
    def adapt(specs, *args, **kwargs):
        specs_seen.extend(specs)
        return [fixed_result(s.model_code, s.seed_index, s.targ).adaptation for s in specs]
    fail_once = True
    def finish(dataset, adapted, spec, config, **kwargs):
        if fail_once and spec.model_code == "ff" and spec.seed_index == 1 and spec.targ == .35:
            raise RuntimeError("simulated save-boundary failure")
        r = fixed_result(spec.model_code, spec.seed_index, spec.targ)
        r.conditions["spec"]["eta"] = spec.eta
        return r
    monkeypatch.setattr(workflows, "run_adaptation_batch", adapt)
    monkeypatch.setattr(workflows, "finish_adaptation", finish)
    def run():
        return workflows.run_selected_experiments(source / "hp/selection", BASE / "dataset.npz", CONFIG,
                                                   tmp_path / "training", batch_runs=2,
                                                   saving=SaveOptions(save_full_test=True))
    with pytest.raises(RuntimeError, match="Some conditions failed"): run()
    assert len(specs_seen) == 12
    selection, hp_specs = workflows.read_selection(source / "hp/selection")
    assert {s.seed_index for s in specs_seen} == {s.seed_index for s in hp_specs}
    for spec in specs_seen:
        assert spec.eta == selection["cells"][f"{spec.model_code}@{spec.targ:g}"]["selected_eta"]
    fail_once = False
    specs_seen.clear()
    states = run()
    assert len(specs_seen) == 1
    assert sorted(states.state).count("skipped") == 11
    assert len(list((tmp_path / "training").rglob("COMPLETE"))) == 12
    specs_seen.clear()
    assert run().state.eq("skipped").all() and specs_seen == []
