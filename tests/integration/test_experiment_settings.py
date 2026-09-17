"""Shared entrypoint settings and their saved source, using the synthetic example."""
import json
from pathlib import Path
import re
import runpy
import sys
from types import ModuleType

import pandas as pd
import pytest
import torch

from hnn2.result_io import write_result

ROOT = Path(__file__).resolve().parents[2]


def settings_for_layout(tmp_path):
    """Import the real path definitions under a temporary root; execute no entry."""
    settings_file = ROOT / "experiment_settings.py"
    settings = ModuleType("experiment_settings")
    settings.__file__ = str(settings_file)
    source = settings_file.read_text(encoding="utf-8")
    source = source.replace('OUTPUT_ROOT = ROOT / "outputs"', f'OUTPUT_ROOT = Path({str(tmp_path)!r})')
    source, replacements = re.subn(r'^EXPERIMENT_NAME = .*$',
                                   'EXPERIMENT_NAME = "layout_check"', source, flags=re.MULTILINE)
    assert replacements == 1
    exec(compile(source, str(settings_file), "exec"), settings.__dict__)
    return settings


def test_settings_paths_without_execution(tmp_path):
    settings = settings_for_layout(tmp_path)
    experiment = tmp_path / "layout_check"
    assert settings.EXPERIMENT_ROOT == experiment
    assert settings.RESULTS_ROOT == experiment / "results"
    assert settings.SWEEP_OUTPUT == experiment / "results/hp/candidates"
    assert settings.SELECTION_OUTPUT == experiment / "results/hp/selection"
    assert settings.EXPERIMENTS_OUTPUT == experiment / "results/plasticity"
    assert settings.BASELINES_OUTPUT == experiment / "results/baselines"
    assert settings.ANALYSIS_OUTPUT == experiment / "analysis" / settings.ANALYSIS_NAME
    assert list(tmp_path.iterdir()) == []


def test_entrypoints_save_shared_settings(tmp_path, monkeypatch):
    # This separate test executes training; do not include it in no-learning checks.
    import matplotlib
    matplotlib.use("Agg")
    original_threads = torch.get_num_threads()
    settings_file = ROOT / "experiment_settings.py"
    settings = settings_for_layout(tmp_path)
    monkeypatch.setitem(sys.modules, "experiment_settings", settings)
    experiment = tmp_path / "layout_check"
    assert settings.EXPERIMENT_ROOT == experiment
    assert not experiment.exists()  # Import creates no output directories.

    def run(name, **overrides):
        entry = ROOT / "scripts" / f"{name}.py"
        main = runpy.run_path(str(entry))["main"]
        for key, value in overrides.items():
            monkeypatch.setitem(main.__globals__, key, value)
        main()

    try:
        run("run_single")
        run("run_baselines")
        run("run_sweep")
        run("run_experiments")
        states = pd.read_csv(experiment / "results/plasticity/last_run.csv")
        sources = [Path(p) for p in states.loc[(states.model_code == "rec") & (states.targ == .35), "path"]]
        run("run_readout", SOURCE=sources[0])
        run("run_analysis", SOURCES=sources,
            RECOMPUTE_METRICS=True, COMPUTE_UMAP=True)
        analysis = experiment / "analysis/comparison_v1"
        assert (analysis / "figures/rate_mean.png").is_file()
        assert (analysis / "figures/rate_mean.pdf").is_file()
        assert (analysis / "tables/monitor/plot_data.csv").is_file()
        assert (analysis / "tables/metrics/entropy.csv").is_file()
        assert (analysis / "arrays/umap/embedding.npz").is_file()
        assert (experiment / "results/hp/selection/selection.json").is_file()
        assert len(list((experiment / "results/hp/candidates").rglob("COMPLETE"))) == len(settings.SWEEP_SPECS)
        assert (experiment / "results/baselines/raw/seed-0/features.npz").is_file()
        assert (experiment / "results/single/rec_target-0.35_seed-0/summary.csv").is_file()
        assert (experiment / "results/readout_variants/lr-0.02_v1/summary.csv").is_file()
        for path in sources:
            assert path.is_relative_to(experiment / "results/plasticity")
            assert (path / "features.npz").is_file() and (path / "summary.csv").is_file()
        assert not (experiment / "COMPLETE").exists()  # No experiment-wide completion claim yet.

        settings_bytes = settings_file.read_bytes()
        snapshots = list(tmp_path.rglob("experiment_settings.py"))
        assert len(snapshots) == len(list(tmp_path.rglob("COMPLETE")))
        for snapshot in snapshots:
            assert snapshot.is_relative_to(experiment)
            assert snapshot.read_bytes() == settings_bytes
            conditions = json.loads((snapshot.parent / "conditions.json").read_text(encoding="utf-8"))
            assert conditions["settings"] == snapshot.name
            assert (snapshot.parent / conditions["script"]).is_file()

        # Skipping complete experiments preserves the original source snapshots.
        before = {p: p.read_bytes() for p in snapshots}
        run("run_pipeline")
        assert all(p.read_bytes() == content for p, content in before.items())
        with pytest.raises(FileExistsError):
            run("run_analysis", SOURCES=sources)
        with pytest.raises(FileExistsError):
            run("run_readout", SOURCE=sources[0])
    finally:
        torch.set_num_threads(original_threads)


def test_missing_settings_source_does_not_create_output(tmp_path):
    output = tmp_path / "result"
    with pytest.raises(FileNotFoundError):
        write_result(output, {}, settings_path=tmp_path / "missing.py")
    assert not output.exists()


def test_failed_settings_copy_is_incomplete_and_can_retry(tmp_path, monkeypatch):
    import hnn2.result_io as result_io
    source = tmp_path / "settings.py"
    source.write_text("# 設定\nVALUE = 1\n", encoding="utf-8")
    output = tmp_path / "result"
    original_atomic = result_io._atomic

    def fail_settings(path, write):
        if path.name == "experiment_settings.py":
            raise OSError("settings copy failed")
        return original_atomic(path, write)

    monkeypatch.setattr(result_io, "_atomic", fail_settings)
    with pytest.raises(OSError, match="settings copy failed"):
        write_result(output, {}, settings_path=source)
    assert not (output / "COMPLETE").exists()
    assert (output / "FAILED.json").is_file()

    monkeypatch.setattr(result_io, "_atomic", original_atomic)
    write_result(output, {}, settings_path=source, resume=True)
    assert (output / "COMPLETE").is_file()
    assert not (output / "FAILED.json").exists()
    assert (output / "experiment_settings.py").read_bytes() == source.read_bytes()

    # Editing the source later never overwrites a completed result.
    original = (output / "experiment_settings.py").read_bytes()
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_result(output, {}, settings_path=source, resume=True)
    assert (output / "experiment_settings.py").read_bytes() == original
