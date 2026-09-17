"""3d-1 regressions: fixed saved numbers and stubbed calls, never learning."""
import json
from pathlib import Path
import shutil
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notebook_cells import table_results, threshold_results
from figure_fixtures import BASE, CONFIG, fixed_experiment, fixed_result
from hnn2.artifact_checks import output_issues, inspect_saved
from hnn2.config import RunSpec
from hnn2.figure_tables import load_figure_runs, save_figure_tables
from hnn2.result_io import SaveOptions, read_single, save_single
from hnn2.readout import train_readout as run_stubbed_readout
import hnn2.workflows as workflows

TARGETS = dict.fromkeys(("rec", "ff", "thresh"), .35)


@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("gap,chosen", [(0., .35), (.5e-6, .35), (1e-6, .35),
                                        (np.nextafter(1e-6, np.inf), .55)])
def test_target_ties_use_absolute_tolerance_and_record_the_chosen_median(maximize, gap, chosen):
    from hnn2.targets import _argmax_table
    worse = -gap if maximize else gap
    frame = pd.DataFrame({"model_code": ["rec"] * 4, "targ": [.55, .35, .55, .35],
                          "score": [0., worse, 0., worse]})
    result = _argmax_table(frame, "score", maximize=maximize)["rec"]
    assert result["chosen_targ"] == chosen
    assert result["value"] == (worse if chosen == .35 else 0.)
    assert result["cells"] == {"0.35": worse, "0.55": 0.}


@pytest.mark.parametrize("maximize", [False, True])
def test_target_tolerance_does_not_grow_with_score_magnitude(maximize):
    from hnn2.targets import _argmax_table
    worse = 100. - 2e-6 if maximize else 100. + 2e-6
    frame = pd.DataFrame({"model_code": ["rec"] * 2, "targ": [.35, .55], "score": [worse, 100.]})
    result = _argmax_table(frame, "score", maximize=maximize)["rec"]
    assert result["chosen_targ"] == .55 and result["value"] == 100.


def test_target_tolerance_compares_to_best_and_retains_unrounded_evidence():
    from hnn2.targets import _argmax_table
    frame = pd.DataFrame({"model_code": ["rec"] * 3, "targ": [.350001, .350002, .55],
                          "score": [1.5e-6, .75e-6, 0.]})
    result = _argmax_table(frame, "score", maximize=False)["rec"]
    assert result["chosen_targ"] == .350002 and result["value"] == .75e-6
    assert result["boundary"] is False
    assert result["cells"] == {"0.350001": 1.5e-6, "0.350002": .75e-6, "0.55": 0.}


@pytest.fixture(autouse=True)
def forbid_learning(monkeypatch):
    import hnn2.single as single
    import hnn2.readout as readout
    import hnn2.mlp as mlp
    import hnn2.umap_embed as umap
    def forbidden(*args, **kwargs):
        pytest.fail("Review tests must not train, infer or generate UMAP")
    for module, names in ((single, ("run_adaptation", "sparse_features", "fit_readout")),
                          (workflows, ("run_adaptation_batch", "finish_adaptation", "run_representation")),
                          (readout, ("train_readout",)), (mlp, ("train_mlp_e2e",)), (umap, ("embed",))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


@pytest.fixture(scope="module")
def fixed_saved(tmp_path_factory):
    return fixed_experiment(tmp_path_factory.mktemp("review_fixed"))


@pytest.mark.parametrize("damage", ["selected", "landscape", "early_epoch"])
def test_hp_checks_all_saved_tables_and_epochs(fixed_saved, tmp_path, damage):
    source = tmp_path / "selection"
    shutil.copytree(fixed_saved / "hp/selection", source)
    name = "curves" if damage == "early_epoch" else damage
    table = pd.read_csv(source / f"{name}.csv", float_precision="round_trip")
    if damage == "early_epoch":
        table = table.drop(table[table.epoch == 0].index[0])
    else:
        table.loc[0, "eta" if name == "selected" else "score"] += .123
    table.to_csv(source / f"{name}.csv", index=False)
    with pytest.raises(ValueError, match="coverage|differs"):
        workflows.read_selection(source)


def test_later_condition_mismatch_is_checked_before_any_training(tmp_path):
    specs = [RunSpec(m, .35, .001, 0) for m in ("rec", "ff")]
    result = fixed_result("ff")
    result.conditions["config"]["readout_lr"] = .123
    save_single(result, workflows.condition_directory(tmp_path, specs[1]))
    with pytest.raises(ValueError, match="Conditions differ"):
        workflows.run_experiments(BASE / "dataset.npz", specs, CONFIG, tmp_path, batch_runs=1)


@pytest.mark.parametrize("damage", ["summary_l1", "history_l1", "batch_accuracy", "feature_width", "rng", "axes", "saved_splits",
                                    "empty_metric", "invalid_silhouette"])
def test_artifact_cross_checks_fixed_values(fixed_saved, damage):
    path = next((fixed_saved / "plasticity").rglob("COMPLETE")).parent
    c, a, t = read_single(path)
    if damage == "summary_l1": t["summary"].loc[0, "final_cumulative_l1"] += 1
    if damage == "history_l1": t["history"].loc[1, "cumulative_l1"] += 1
    if damage == "batch_accuracy": t["history"].loc[0, "batch_accuracy"] = 1.5
    if damage == "feature_width":
        t["summary"].loc[0, "n_features"] = CONFIG.n_excitatory - 1
        for key in list(a["features"]):
            if key.startswith("h_"): a["features"][key] = a["features"][key][:, :-1]
    if damage == "rng": c["rng"]["streams"]["shuffle"] = 99
    if damage == "axes": c["axes"]["activity"] = "samples_units"
    if damage == "saved_splits": c["saved_feature_splits"] = ["train"]
    if damage == "empty_metric": t["entropy"] = t["entropy"].iloc[:0]
    if damage == "invalid_silhouette": t["silhouette"].loc[0, "silhouette"] = 2.
    assert output_issues(c, a, t)


@pytest.mark.parametrize("table,column", [("entropy", "axis"), ("silhouette", "metric")])
def test_missing_metric_column_is_reported_as_invalid(fixed_saved, tmp_path, table, column):
    source = next((fixed_saved / "plasticity").rglob("COMPLETE")).parent
    path = tmp_path / "run"
    shutil.copytree(source, path)
    data = pd.read_csv(path / f"{table}.csv").drop(columns=column)
    data.to_csv(path / f"{table}.csv", index=False)
    assert inspect_saved(path)


@pytest.mark.parametrize("damage", ["missing_target", "unknown_target", "missing_seed", "duplicate"])
def test_threshold_population_cannot_silently_shrink(fixed_saved, damage):
    runs, expected = load_figure_runs(fixed_saved)
    targets = TARGETS.copy()
    if damage == "missing_target": targets.pop("rec")
    if damage == "unknown_target": targets["rec"] = .999
    if damage == "missing_seed": expected = expected.drop(expected[expected.arm == "raw"].index[0])
    if damage == "duplicate": expected = pd.concat([expected, expected.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="target|population|seed|Duplicate"):
        threshold_results(runs, expected, targets)


def test_threshold_incomplete_observations_are_not_unreached(fixed_saved):
    runs, expected = load_figure_runs(fixed_saved)
    path = expected[expected.arm == "raw"].run.iloc[0]
    runs[path][2]["history"] = runs[path][2]["history"].drop(index=1)
    detail, summary = threshold_results(runs, expected, TARGETS)
    assert detail[detail.run == path].judgement.eq("invalid").all()
    assert summary[summary.arm == "raw"].missing_or_invalid.eq(1).all()


def test_threshold_does_not_require_other_figure_artifacts(fixed_saved, tmp_path):
    root = tmp_path / "results"
    shutil.copytree(fixed_saved, root)
    # Rewrite only temporary planned paths after copying, then omit unrelated outputs.
    for name in ("plasticity", "baselines"):
        file = root / name / "expected_runs.csv"
        pop = pd.read_csv(file)
        pop["path"] = pop.path.map(lambda p: str(root / Path(p).relative_to(fixed_saved)))
        pop.to_csv(file, index=False)
    for mark in root.rglob("COMPLETE"):
        if "hp" in mark.parts: continue
        cfile = mark.parent / "conditions.json"
        c = json.loads(cfile.read_text(encoding="utf-8"))
        for key in ("save_features", "save_full_test", "save_encoder_weights", "save_activity", "save_summaries"):
            c[key] = False
        c["saved_feature_splits"] = []
        cfile.write_text(json.dumps(c), encoding="utf-8")
        for file in mark.parent.glob("*.npz"): file.unlink()
    runs, expected = load_figure_runs(root, readout_only=True)
    _, summary = threshold_results(runs, expected, TARGETS)
    assert summary.total.eq(2).all() and summary.missing_or_invalid.eq(0).all()
    with pytest.raises(ValueError, match="figure input|unavailable|incomplete"):
        table_results(runs, expected, root / "hp/selection", TARGETS)


@pytest.mark.parametrize("damage", ["selection", "population", "targets", "complete"])
def test_derived_output_refuses_changed_provenance(fixed_saved, tmp_path, damage):
    root = tmp_path / "results"
    shutil.copytree(fixed_saved, root)
    for name in ("plasticity", "baselines"):
        file = root / name / "expected_runs.csv"
        pop = pd.read_csv(file)
        pop["path"] = pop.path.map(lambda p: str(root / Path(p).relative_to(fixed_saved)))
        pop.to_csv(file, index=False)
    runs, expected = load_figure_runs(root)
    tables = table_results(runs, expected, root / "hp/selection", TARGETS)
    targets = TARGETS.copy()
    if damage == "targets": targets["rec"] = .55
    if damage in ("selection", "population"):
        path = root / ("hp/selection/selection.json" if damage == "selection" else "plasticity/expected_runs.csv")
        path.write_bytes(path.read_bytes() + b"\n")
    if damage == "complete": (Path(next(iter(runs))) / "COMPLETE").unlink()
    with pytest.raises(ValueError, match="changed|differ|Source"):
        save_figure_tables(tmp_path / "derived", tables, runs, targets=targets, selection_directory=root / "hp/selection")
    assert not (tmp_path / "derived").exists()


def test_mlp_learning_budget_cannot_mix_between_seeds(fixed_saved, tmp_path):
    root = tmp_path / "results"
    shutil.copytree(fixed_saved, root)
    for name in ("plasticity", "baselines"):
        file = root / name / "expected_runs.csv"
        pop = pd.read_csv(file)
        pop["path"] = pop.path.map(lambda p: str(root / Path(p).relative_to(fixed_saved)))
        pop.to_csv(file, index=False)
    file = root / "baselines/mlp_frozen/seed-1/conditions.json"
    c = json.loads(file.read_text(encoding="utf-8"))
    c["mlp"]["epochs"] += 1
    file.write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(ValueError, match="MLP/readout conditions differ"):
        load_figure_runs(root)


def test_silhouette_one_sample_per_class_is_undefined():
    from hnn2.metrics.silhouette import representation_silhouette
    assert representation_silhouette(np.eye(3), np.arange(3)) is None


@pytest.mark.parametrize("baseline", [False, True])
def test_impossible_saving_request_stops_before_work(tmp_path, baseline):
    config = replace(CONFIG, readout_final_eval_full_test=False)
    with pytest.raises(ValueError, match="Full-test features will not be computed"):
        if baseline:
            workflows.run_baselines(BASE / "dataset.npz", ("rec",), (0,), config, tmp_path,
                                    saving=SaveOptions(save_full_test=True))
        else:
            workflows.run_experiments(BASE / "dataset.npz", [RunSpec("rec", .35, .001, 0)], config,
                                      tmp_path, saving=SaveOptions(save_full_test=True))
    assert not list(tmp_path.iterdir())


def test_nonempty_readout_log_uses_existing_evaluation_order(monkeypatch):
    """Exercise logging with fixed logits/loss and a counting optimizer, no training."""
    import torch
    import hnn2.readout as readout
    events, evaluations = [], []
    class Model:
        def __init__(self, width, classes):
            self.weight, self.bias = torch.zeros(classes, width), torch.zeros(classes)
            self.steps = 0
        def parameters(self): return [self.weight, self.bias]
        def to(self, device): return self
        def train(self): return self
        def __call__(self, x):
            events.append("fixed_logits")
            return torch.zeros(len(x), len(self.bias))
    class Loss:
        def backward(self): events.append("stub_backward")
        def __float__(self): return .5
    class Optimizer:
        def zero_grad(self): pass
        def step(self):
            events.append("stub_step")
            model.steps += 1
    model = Model(CONFIG.n_excitatory, 3)
    monkeypatch.setattr(readout.nn, "Linear", lambda *args: model)
    monkeypatch.setattr(readout.nn, "CrossEntropyLoss", lambda: lambda *args: Loss())
    monkeypatch.setattr(readout.torch.optim, "Adam", lambda *args, **kwargs: Optimizer())
    def evaluate(*args):
        evaluations.append(model.steps)
        return 1 / (1 + model.steps), .3 + .05 * model.steps
    monkeypatch.setattr(readout, "_evaluate", evaluate)
    features = fixed_result().features
    result = run_stubbed_readout(features, features, CONFIG, init_seed=1, shuffle_seed=2, device=torch.device("cpu"))
    assert events == ["fixed_logits", "stub_backward", "stub_step"] * 9
    assert evaluations == [2, 4, 6, 8, 9, 9, 9]  # periodic, final val, test, full test
    assert result.history.val_evaluated == [False, True, False, True, False, True, False, True, False]
    assert result.final_step == 9 and result.final_val_accuracy == .75
    assert result.history.val_accuracy[-1] == .7


@pytest.mark.parametrize("representation,model", [("sparse_initial", m) for m in ("rec", "ff", "thresh")]
                         + [("raw", "rec"), ("mlp_frozen", "rec")])
def test_each_baseline_calls_its_classifier_once_with_the_recorded_stream(monkeypatch, representation, model):
    import torch
    import hnn2.single as single
    import hnn2.mlp as mlp
    import hnn2.model.init as init
    frozen = fixed_result(model, 1, representation=representation)
    features = {k: v for k, v in frozen.features.items() if k.startswith("h_")}
    calls = []
    class Encoder:
        def to_arrays(self): return frozen.encoder_weights
        def state_dict(self):
            return {k: torch.as_tensor(frozen.encoder_weights[k])
                    for k in ("hidden.weight", "hidden.bias", "readout.weight", "readout.bias")}
    def build(code, config, seed, width, device):
        calls.append(("initial", code, seed))
        return Encoder()
    def train(*args, **kwargs):
        from hnn2.rng import stream_torch_seed, stream
        assert kwargs["init_seed"] == stream_torch_seed(1, "mlp/init")
        assert kwargs["shuffle_seed"] == int(stream(1, "mlp/shuffle").integers(0, 2**31))
        calls.append(("mlp", args[2].readout_epochs))
        return Encoder()
    def fit(values, config, seed, **kwargs):
        calls.append(("classifier", seed, kwargs["init_stream"]))
        return frozen.readout, {}
    monkeypatch.setattr(init, "build_params", build)
    monkeypatch.setattr(mlp, "train_mlp_e2e", train)
    monkeypatch.setattr(single, "sparse_features", lambda *args, **kwargs: dict(features))
    monkeypatch.setattr(single, "mlp_frozen_features", lambda *args, **kwargs: dict(features))
    monkeypatch.setattr(single, "fit_readout", fit)
    settings = single.MlpSettings(epochs=2)
    result = single.run_representation(BASE / "dataset.npz", CONFIG, 1, representation=representation,
                                       model_code=model, mlp=settings)
    expected_stream = "mlp/readout_init" if representation == "mlp_frozen" else "readout/init"
    assert calls[-1] == ("classifier", 1, expected_stream)
    assert len([c for c in calls if c[0] == "classifier"]) == 1
    assert result.conditions["readout_streams"] == {"init": expected_stream, "shuffle": "readout/shuffle"}
    if representation == "sparse_initial": assert calls[0] == ("initial", model, 1)
    if representation == "mlp_frozen": assert calls[0] == ("mlp", settings.epochs)


def test_hp_seed_mismatch_rejected_without_learning(fixed_saved, tmp_path):
    selection, specs = workflows.read_selection(fixed_saved / "hp/selection")
    from hnn2.config import experiment_config_from_json
    from hnn2.hp.rules import SelectionRule
    specs = [s for s in specs if not (s.model_code == "rec" and s.targ == .35 and s.seed_index == 1)]
    curves = pd.read_csv(fixed_saved / "hp/selection/curves.csv", float_precision="round_trip")
    curves = curves[~((curves.model_code == "rec") & (curves.targ == .35) & (curves.seed_index == 1))]
    workflows.select_sweep(curves, specs, experiment_config_from_json(json.dumps(selection["sweep_config"])),
                           SelectionRule(**selection["rule"]), tmp_path / "selection", training_epochs=CONFIG.n_epochs,
                           input_source=selection["input"])
    with pytest.raises(ValueError, match="same seed set"):
        workflows.run_selected_experiments(tmp_path / "selection", BASE / "dataset.npz", CONFIG, tmp_path / "training")
    assert not (tmp_path / "training").exists()


def test_nonvalidation_monitor_is_not_reported_as_validation_evidence(fixed_saved, tmp_path):
    from hnn2.config import experiment_config_from_json
    from hnn2.hp.rules import SelectionRule
    selection, specs = workflows.read_selection(fixed_saved / "hp/selection")
    sweep = replace(experiment_config_from_json(json.dumps(selection["sweep_config"])), monitor_probe_split="train")
    curves = pd.read_csv(fixed_saved / "hp/selection/curves.csv", float_precision="round_trip")
    workflows.select_sweep(curves, specs, sweep, SelectionRule(**selection["rule"]), tmp_path / "selection",
                           training_epochs=CONFIG.n_epochs, input_source=selection["input"])
    runs, expected = load_figure_runs(fixed_saved)
    for c, _, _ in runs.values(): c["config"]["monitor_probe_split"] = "train"
    with pytest.raises(ValueError, match="requires validation monitor"):
        table_results(runs, expected, tmp_path / "selection", TARGETS)


def test_expected_run_paths_are_unique_after_resolution(tmp_path):
    rows = [{"representation": "raw", "model_code": "", "targ": None, "eta": None, "seed_index": seed,
             "path": str(path)} for seed, path in enumerate((tmp_path / "run", tmp_path / "other/../run"))]
    with pytest.raises(ValueError, match="unique"):
        workflows.save_expected_runs(tmp_path, rows, config=CONFIG)
    assert not (tmp_path / "expected_runs.csv").exists()
