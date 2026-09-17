"""Storage switches preserve numbers and distinguish omission from damage."""
from dataclasses import replace
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest
import torch

from hnn2.config import ExperimentConfig, RunSpec
from hnn2.postprocess import saved_features, readout_only, features_from_encoder, compare_tables
from hnn2.result_io import SaveOptions, read_single, readout_tables, save_single
from hnn2.single import MlpSettings, run_single, run_representation
from hnn2.workflows import run_experiments

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/fixtures/m3_reference"
FROZEN = json.loads((BASE / "conditions.json").read_text(encoding="utf-8"))
CONFIG = ExperimentConfig(**{k: tuple(v) if k in ("image_size", "raw_baseline_size") else v
                            for k, v in FROZEN["config"].items()})
SPEC = RunSpec("rec", .35, .001, 0)
ALL = dict(save_features=True, save_full_test=True, save_encoder_weights=True,
           save_activity=True, save_summaries=True, save_readout_history=True)
NONE = dict.fromkeys(ALL, False)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture(scope="module")
def results():
    values = {"sparse_trained": run_single(BASE / "dataset.npz", SPEC, CONFIG)}
    for representation in ("sparse_initial", "raw", "mlp_frozen"):
        values[representation] = run_representation(
            BASE / "dataset.npz", CONFIG, 0, representation=representation,
            mlp=MlpSettings(**FROZEN["mlp"]))
    return values


@pytest.mark.parametrize("options", [ALL, NONE, *[{**ALL, key: False} for key in ALL],
                                    {**ALL, "save_features": False, "save_full_test": False}])
def test_each_switch_retains_identical_remaining_numbers(results, tmp_path, options):
    result = results["sparse_trained"]
    full = save_single(result, tmp_path / "full", saving=SaveOptions(**ALL))
    path = save_single(result, tmp_path / "selected", saving=SaveOptions(**options))
    conditions, arrays, tables = read_single(path)
    _, full_arrays, full_tables = read_single(full)
    assert all(conditions[key] == value for key, value in options.items())
    expected = {"weights_initial", "weights_trained"} if options["save_encoder_weights"] else set()
    expected.update(name for name in ("summaries", "activity") if options[f"save_{name}"])
    splits = (["train", "val", "test"] if options["save_features"] else [])
    splits += ["test_full"] if options["save_full_test"] else []
    if splits:
        expected.add("features")
    assert set(arrays) == expected
    assert {p.stem for p in path.glob("*.npz")} == expected
    assert conditions["saved_feature_splits"] == splits
    for name, values in arrays.items():
        keys = {f"{prefix}_{split}" for prefix in ("h", "y", "sample_index") for split in splits} if name == "features" else set(full_arrays[name])
        assert set(values) == keys
        for key, value in values.items():
            np.testing.assert_array_equal(value, full_arrays[name][key])
    expected_tables = {"summary", "monitor", "entropy", "silhouette"}
    if options["save_readout_history"]:
        expected_tables.add("history")
    assert set(tables) == expected_tables
    assert {p.stem for p in path.glob("*.csv")} == expected_tables
    for name, table in tables.items():
        pd.testing.assert_frame_equal(table, full_tables[name], check_exact=True)
    assert tables["summary"].final_test_accuracy_full.notna().all()


@pytest.mark.parametrize("representation", ["sparse_initial", "raw", "mlp_frozen"])
def test_baseline_weights_and_reextraction(results, tmp_path, representation):
    result = results[representation]
    weights = save_single(result, tmp_path / "weights", saving=SaveOptions(**{**NONE, "save_encoder_weights": True}))
    _, arrays, _ = read_single(weights)
    assert set(arrays) == (set() if representation == "raw" else {"encoder_weights"})
    restored = features_from_encoder(weights, BASE / "dataset.npz", CONFIG)
    for key in restored:
        np.testing.assert_array_equal(restored[key], result.features[key])
    minimal = save_single(result, tmp_path / "minimal", saving=SaveOptions(**NONE))
    assert read_single(minimal)[1] == {}
    if representation != "raw":
        with pytest.raises(ValueError, match="weights were intentionally not saved"):
            features_from_encoder(minimal, BASE / "dataset.npz", CONFIG)


def test_postprocessing_omission_and_damage(results, tmp_path):
    result = results["sparse_trained"]
    weights = save_single(result, tmp_path / "weights", saving=SaveOptions(**{**NONE, "save_encoder_weights": True}))
    with pytest.raises(ValueError, match="intentionally not saved"):
        readout_only(weights, CONFIG, tmp_path / "no_readout")
    restored = features_from_encoder(weights, BASE / "dataset.npz", CONFIG)
    for key in restored:
        np.testing.assert_array_equal(restored[key], result.features[key])
    full_only = save_single(result, tmp_path / "full_only", saving=SaveOptions(**{**NONE, "save_full_test": True}))
    saved_features(full_only, splits=["test_full"])
    with pytest.raises(ValueError, match="train features were intentionally not saved"):
        readout_only(full_only, CONFIG, tmp_path / "no_training_features")
    no_weights = save_single(result, tmp_path / "features", saving=SaveOptions(**{**ALL, "save_encoder_weights": False}))
    with pytest.raises(ValueError, match="weights were intentionally not saved"):
        features_from_encoder(no_weights, BASE / "dataset.npz", CONFIG)
    with pytest.raises(ValueError, match="history was intentionally not saved"):
        compare_tables([weights], "history")
    (no_weights / "features.npz").unlink()
    with pytest.raises(ValueError, match="Incomplete.*features"):
        read_single(no_weights)
    with pytest.raises(FileNotFoundError):
        saved_features(no_weights)


def test_readout_history_does_not_change_final_evaluation(results, tmp_path):
    source = save_single(results["sparse_trained"], tmp_path / "source", saving=SaveOptions(**ALL))
    first = readout_only(source, CONFIG, tmp_path / "with_history")
    second = readout_only(source, CONFIG, tmp_path / "without_history", save_readout_history=False)
    for name, table in readout_tables(first).items():
        pd.testing.assert_frame_equal(table, readout_tables(second)[name], check_exact=True)
    assert not (tmp_path / "without_history/history.csv").exists()
    assert (tmp_path / "with_history/summary.csv").read_bytes() == (tmp_path / "without_history/summary.csv").read_bytes()
    with pytest.raises(ValueError, match="history was intentionally not saved"):
        compare_tables([tmp_path / "without_history"], "history")


def test_completed_minimal_runs_skip_but_damaged_required_files_fail(tmp_path, monkeypatch):
    import hnn2.workflows as workflows
    path = tmp_path / "runs"
    first = run_experiments(BASE / "dataset.npz", [SPEC], CONFIG, path, saving=SaveOptions(**NONE))
    assert first.state.tolist() == ["complete"]
    reference = run_experiments(BASE / "dataset.npz", [SPEC], CONFIG, tmp_path / "all", saving=SaveOptions(**ALL))
    _, _, minimal_tables = read_single(first.path.iloc[0])
    _, _, reference_tables = read_single(reference.path.iloc[0])
    for name, table in minimal_tables.items():
        pd.testing.assert_frame_equal(table, reference_tables[name], check_exact=True)
    def forbidden(*args, **kwargs):
        pytest.fail("Complete results must not be recomputed")
    monkeypatch.setattr(workflows, "run_adaptation_batch", forbidden)
    second = run_experiments(BASE / "dataset.npz", [SPEC], CONFIG, path, saving=SaveOptions(**NONE))
    assert second.state.tolist() == ["skipped"]
    with pytest.raises(ValueError, match="Conditions differ"):
        run_experiments(BASE / "dataset.npz", [SPEC], CONFIG, path, saving=SaveOptions(**ALL))
    (Path(first.path.iloc[0]) / "summary.csv").unlink()
    with pytest.raises(ValueError, match="Completed result lacks.*summary|Incomplete single result.*summary"):
        run_experiments(BASE / "dataset.npz", [SPEC], CONFIG, path, saving=SaveOptions(**NONE))


def test_incomplete_save_requires_same_storage_choices(results, tmp_path, monkeypatch):
    import hnn2.result_io as storage
    original = storage._npz
    def failure(*args):
        raise OSError("disk error")
    monkeypatch.setattr(storage, "_npz", failure)
    path = tmp_path / "partial"
    with pytest.raises(OSError):
        save_single(results["sparse_trained"], path, saving=SaveOptions(**ALL))
    assert not (path / "COMPLETE").exists()
    with pytest.raises(ValueError, match="different conditions"):
        save_single(results["sparse_trained"], path, saving=SaveOptions(**NONE), resume=True)
    monkeypatch.setattr(storage, "_npz", original)
    save_single(results["sparse_trained"], path, saving=SaveOptions(**ALL), resume=True)
    read_single(path)


# Missing saving/stream records are now rejected. The no-learning cases live in
# test_figure_artifacts.py; old fixture values remain unchanged.


def test_requested_full_features_must_have_been_computed(results, tmp_path):
    result = results["sparse_trained"]
    partial = replace(result, features={key: value for key, value in result.features.items() if "test_full" not in key})
    with pytest.raises(ValueError, match="Full-test features were not computed"):
        save_single(partial, tmp_path / "not_created", saving=SaveOptions(**ALL))
    assert not (tmp_path / "not_created").exists()


def test_promised_feature_split_cannot_silently_disappear(results, tmp_path):
    path = save_single(results["sparse_trained"], tmp_path / "damaged", saving=SaveOptions(**ALL))
    arrays = {key: value for key, value in results["sparse_trained"].features.items() if key != "y_test_full"}
    np.savez_compressed(path / "features.npz", **arrays)
    with pytest.raises(ValueError, match="y_test_full"):
        saved_features(path)


def run_entry(name, monkeypatch, **overrides):
    main = runpy.run_path(str(ROOT / "scripts" / f"{name}.py"))["main"]
    for key, value in overrides.items():
        monkeypatch.setitem(main.__globals__, key, value)
    main()


def test_all_training_entrypoints_forward_switches(tmp_path, monkeypatch):
    disabled = {"SAVING": SaveOptions(**NONE)}
    run_entry("run_single", monkeypatch, OUTPUT=tmp_path / "single", **disabled)
    run_entry("run_baselines", monkeypatch, OUTPUT=tmp_path / "baseline", **disabled)
    run_entry("run_pipeline", monkeypatch, SWEEP_OUTPUT=tmp_path / "sweep",
              SELECTION_OUTPUT=tmp_path / "selection", EXPERIMENTS_OUTPUT=tmp_path / "trained",
              BASELINES_OUTPUT=tmp_path / "baseline", **disabled)
    # Train-only must skip these minimal results using the identical policy.
    run_entry("run_experiments", monkeypatch, SOURCE=tmp_path / "selection", OUTPUT=tmp_path / "trained", **disabled)
    states = pd.read_csv(tmp_path / "trained/last_run.csv")
    assert states.state.eq("skipped").all()
    assert not list(tmp_path.rglob("*.npz"))
    assert not list(tmp_path.rglob("history.csv"))
    assert len(list(tmp_path.rglob("summary.csv"))) == 23  # single + 10 Baselines + 12 trained
    assert list(tmp_path.rglob("curves.csv"))  # HP evidence always stays.


@pytest.mark.parametrize("png,pdf", [(False, False), (True, False), (False, True), (True, True)])
def test_figure_switches_are_independent_and_keep_plot_numbers(results, tmp_path, monkeypatch, png, pdf):
    import matplotlib
    matplotlib.use("Agg")
    source = save_single(results["sparse_trained"], tmp_path / "source", saving=SaveOptions(**NONE))
    output = tmp_path / "plot"
    run_entry("run_analysis", monkeypatch, SOURCES=[source], OUTPUT=output,
              SAVE_FIGURE_PNG=png, SAVE_FIGURE_PDF=pdf)
    assert (output / "figures/rate_mean.png").exists() == png
    assert (output / "figures/rate_mean.pdf").exists() == pdf
    assert (output / "figures").exists() == (png or pdf)
    assert (output / "tables/monitor/plot_data.csv").is_file()
    assert (output / "tables/monitor/COMPLETE").is_file()
    assert not (output / "arrays").exists()


def test_readout_entrypoint_forwards_history_switch(results, tmp_path, monkeypatch):
    source = save_single(results["sparse_trained"], tmp_path / "source", saving=SaveOptions(**ALL))
    run_entry("run_readout", monkeypatch, SOURCE=source, OUTPUT=tmp_path / "readout", SAVE_READOUT_HISTORY=False)
    assert (tmp_path / "readout/summary.csv").is_file()
    assert not (tmp_path / "readout/history.csv").exists()
