"""Migration checks against results frozen from the original Task implementation."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from hnn2.config import ExperimentConfig, RunSpec
from hnn2.single import run_single
from hnn2.result_io import SaveOptions, read_single, save_single

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/fixtures/m2_reference"
CONDITIONS = json.loads((BASE / "conditions.json").read_text(encoding="utf-8"))
CONFIG = ExperimentConfig(**{key: tuple(value) if key in ("image_size", "raw_baseline_size") else value
                            for key, value in CONDITIONS["config"].items()})
SPEC = RunSpec(**CONDITIONS["spec"])


def old_arrays(name):
    with np.load(BASE / "reference" / f"{name}.npz", allow_pickle=False) as source:
        return {key: source[key] for key in source.files if key != "meta"}


def equal_arrays(actual, expected):
    # Additive sample-row coordinates are validated by test_figure_artifacts.
    actual = {k: v for k, v in actual.items() if k != "sample_index" and not k.startswith("sample_index_")}
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].dtype == expected[key].dtype, key
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def equal_table(actual, name):
    expected = pd.read_csv(BASE / "reference" / f"{name}.csv", float_precision="round_trip")
    additions = {"summary": ["final_val_loss", "final_step"], "history": ["val_evaluated"]}
    actual = actual.drop(columns=additions.get(name, []), errors="ignore")
    tolerance = CONDITIONS["comparison"]["csv"]
    pd.testing.assert_frame_equal(actual, expected, check_exact=False,
                                  rtol=tolerance["rtol"], atol=tolerance["atol"])


@pytest.fixture(scope="module")
def result():
    threads = torch.get_num_threads()
    torch.set_num_threads(CONDITIONS["torch_threads"])
    try:
        yield run_single(BASE / "dataset.npz", SPEC, CONFIG)
    finally:
        torch.set_num_threads(threads)


def test_frozen_reference_and_input_are_unchanged():
    for relative, digest in json.loads((BASE / "reference_hashes.json").read_text()).items():
        assert hashlib.sha256((BASE / relative).read_bytes()).hexdigest() == digest


def test_direct_arrays_match_original_tasks_exactly(result):
    equal_arrays(result.adaptation.params_initial.to_arrays(), old_arrays("weights_initial"))
    equal_arrays(result.adaptation.params_trained.to_arrays(), old_arrays("weights_trained"))
    equal_arrays(result.adaptation.summaries_arrays(), old_arrays("summaries"))
    activity = {f"o_E_ep{epoch:02d}": values for epoch, values in result.adaptation.analysis_activity.items()}
    activity["labels"] = result.features["y_test"]
    equal_arrays(activity, old_arrays("activity"))
    equal_arrays(result.features, old_arrays("features"))
    assert np.count_nonzero(result.features["h_test"]) > 0


def test_direct_measurements_match_original_tasks(result):
    equal_table(pd.DataFrame(result.adaptation.monitor_records), "monitor")
    equal_table(pd.DataFrame(asdict(result.readout.history)), "history")
    summary = {key: value for key, value in asdict(result.readout).items() if key != "history"}
    summary["final_cumulative_l1"] = result.readout.history.cumulative_l1[-1]
    equal_table(pd.DataFrame([summary]), "summary")
    equal_table(result.entropy, "entropy")
    equal_table(result.silhouette, "silhouette")
    assert result.conditions["seeds"] == CONDITIONS["seeds"]


def test_save_and_read_need_no_scientific_recomputation(result, tmp_path, monkeypatch):
    import hnn2.single as single
    def forbidden(*args, **kwargs):
        pytest.fail("Saving/reading must not run a scientific computation")
    monkeypatch.setattr(single, "run_adaptation", forbidden)
    monkeypatch.setattr(single, "train_readout", forbidden)
    out = save_single(result, tmp_path / "run", script_path=ROOT / "scripts/run_single.py")
    conditions, arrays, tables = read_single(out)
    assert conditions["spec"] == CONDITIONS["spec"]
    assert conditions["config"] == CONDITIONS["config"]
    assert conditions["input"]["sha256"] == hashlib.sha256((BASE / "dataset.npz").read_bytes()).hexdigest()
    assert conditions["saved_feature_splits"] == ["train", "val", "test"]
    assert "h_test_full" not in arrays["features"]
    assert tables["summary"].final_test_accuracy_full.notna().all()
    assert (out / "experiment.py").read_bytes() == (ROOT / "scripts/run_single.py").read_bytes()
    for name in ("weights_initial", "weights_trained", "summaries", "activity"):
        equal_arrays(arrays[name], old_arrays(name))
    equal_arrays(arrays["features"], {k: v for k, v in result.features.items() if "test_full" not in k})
    for name, table in tables.items():
        equal_table(table, name)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        save_single(result, out)
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}


def test_full_features_can_be_explicitly_retained(result, tmp_path):
    out = save_single(result, tmp_path / "full", saving=SaveOptions(save_full_test=True))
    conditions, arrays, _ = read_single(out)
    equal_arrays(arrays["features"], old_arrays("features"))
    assert "test_full" in conditions["saved_feature_splits"]


def test_failed_save_is_not_complete(result, tmp_path, monkeypatch):
    import hnn2.result_io as storage
    def disk_error(*args):
        raise OSError("simulated disk write failure")
    monkeypatch.setattr(storage, "_npz", disk_error)
    out = tmp_path / "failed"
    with pytest.raises(OSError, match="simulated"):
        save_single(result, out)
    assert not (out / "COMPLETE").exists()
    assert json.loads((out / "FAILED.json").read_text())["error"] == "OSError"
    with pytest.raises(ValueError, match="Incomplete"):
        read_single(out)


@pytest.mark.parametrize("damage", ["labels", "width", "full", "nonfinite"])
def test_invalid_inputs_stop_before_adaptation(tmp_path, monkeypatch, damage):
    import hnn2.single as single
    with np.load(BASE / "dataset.npz", allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files if key != "meta"}
    if damage == "labels":
        arrays["y_train"] = arrays["y_train"][:-1]
    elif damage == "width":
        arrays["x_val_small"] = arrays["x_val_small"][:, :-1]
    elif damage == "full":
        del arrays["x_test_full_small"]
    else:
        arrays["x_train_small"][0, 0] = np.nan
    dataset = tmp_path / "invalid.npz"
    np.savez(dataset, **arrays)
    def forbidden(*args):
        pytest.fail("Invalid data reached adaptation")
    monkeypatch.setattr(single, "run_adaptation", forbidden)
    with pytest.raises(ValueError):
        run_single(dataset, SPEC, CONFIG)


def test_new_entrypoint_has_no_framework_imports_or_import_side_effects(tmp_path):
    script = """
import importlib.abc, runpy, sys
class NoFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in
               ('hnn2.execution', 'hnn2.experiments', 'hnn2.report', 'hnn2.artifact_schema', 'hnn2._source_guard')):
            raise AssertionError('Unexpected framework import: ' + fullname)
sys.meta_path.insert(0, NoFramework())
from pathlib import Path
for entry in Path(sys.argv[1]).glob('*.py'):
    runpy.run_path(str(entry))
"""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run([sys.executable, "-c", script, str(ROOT / "scripts")],
                               cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []
