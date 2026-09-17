"""Shared checks and chapter-only exports, using fixed saved arrays only."""
import copy
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_fixtures import fixed_experiment
import hnn2.figure_tables as figures
from notebook_cells import SCORE_ARMS, SCORE_TABLES, frozen_reference, table_results


@pytest.fixture(autouse=True)
def forbid_learning(monkeypatch):
    import hnn2.single as single
    import hnn2.workflows as workflows
    import hnn2.mlp as mlp
    import hnn2.umap_embed as umap
    def forbidden(*args, **kwargs):
        pytest.fail("No learning, inference or UMAP in boundary tests")
    for module, names in ((single, ("run_adaptation", "fit_readout", "sparse_features")),
                          (workflows, ("run_adaptation_batch", "run_representation")),
                          (mlp, ("train_mlp_e2e",)), (umap, ("embed",))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    return fixed_experiment(tmp_path_factory.mktemp("figure_boundaries"))


@pytest.mark.parametrize("targets", [None, {"rec": .35, "ff": .55, "thresh": .35}])
def test_tables_preserve_snapshot_values_for_the_requested_arms(saved, targets):
    previous = frozen_reference()
    runs, expected = figures.load_figure_runs(saved)
    before = previous.make_figure_tables(runs, expected, saved / "hp/selection", targets)
    after = table_results(runs, expected, saved / "hp/selection", targets)
    assert set(before) == set(after) and len(after) == 36
    for name in before:
        wanted = before[name]
        if name in SCORE_TABLES:
            wanted = wanted[wanted.arm.isin(SCORE_ARMS)].reset_index(drop=True)
        pd.testing.assert_frame_equal(wanted, after[name], check_exact=True, obj=name)
    assert before["coverage"].attrs == after["coverage"].attrs


def test_one_chapter_export_without_other_chapters(saved, tmp_path, monkeypatch):
    runs, expected = figures.load_figure_runs(saved)
    joined = frozen_reference().joined_tables(runs)
    def forbidden(*args, **kwargs):
        pytest.fail("A chapter must not compute every other chapter's tables")
    for name in ("activity_tables", "endpoint_anchors"):
        monkeypatch.setattr(figures, name, forbidden)
    candidates, effective = figures.analysis_target_tables(joined["summary"])
    targets = effective.set_index("model_code").targ.to_dict()
    report, coverage = figures.check_figure_inputs(runs, expected, saved / "hp/selection", targets)
    assert report["cells"] and len(candidates) == 3
    assert "figure_sources" not in expected.attrs  # The caller's coverage is not mutated.
    path = figures.save_figure_tables(tmp_path / "adaptation", {"monitor": joined["monitor"]}, runs,
        targets=targets, selection_directory=saved / "hp/selection",
        coverage=coverage, analysis_targets=effective)
    assert {p.name for p in path.glob("*.csv")} == {"monitor.csv"}
    pd.testing.assert_frame_equal(pd.read_csv(path / "monitor.csv", float_precision="round_trip"),
                                  joined["monitor"], check_exact=True)
    conditions = json.loads((path / "conditions.json").read_text(encoding="utf-8"))
    assert len(conditions["source"]) == len(runs) == 22
    assert conditions["analysis_targets"] == effective.to_dict("records")
    assert conditions["display_targets"] == targets
    assert conditions["selection"] == coverage.attrs["figure_sources"]["selection"]
    assert conditions["population_sources"] == expected.attrs["population_sources"]
    assert (path / "COMPLETE").is_file()
    with pytest.raises(FileExistsError):
        figures.save_figure_tables(path, {"monitor": joined["monitor"]}, runs,
            targets=targets, selection_directory=saved / "hp/selection",
            coverage=coverage, analysis_targets=effective)


@pytest.mark.parametrize("damage", ["run_file", "selection_file", "population_file", "targets", "metadata"])
def test_chapter_export_keeps_source_and_target_checks(saved, tmp_path, damage):
    root = tmp_path / "results"
    shutil.copytree(saved, root)
    for name in ("plasticity", "baselines"):
        path = root / name / "expected_runs.csv"
        population = pd.read_csv(path)
        population["path"] = population.path.map(lambda p: str(root / Path(p).relative_to(saved)))
        population.to_csv(path, index=False)
    runs, expected = figures.load_figure_runs(root)
    summary = frozen_reference().joined_tables(runs)["summary"]
    _, effective = figures.analysis_target_tables(summary)
    targets = effective.set_index("model_code").targ.to_dict()
    _, coverage = figures.check_figure_inputs(runs, expected, root / "hp/selection", targets)
    if damage == "targets":
        targets["rec"] = .55
    elif damage == "metadata":
        effective.loc[effective.model_code == "rec", "targ"] = .55
    else:
        path = {"run_file": Path(next(iter(runs))) / "COMPLETE",
                "selection_file": root / "hp/selection/selection.json",
                "population_file": root / "plasticity/expected_runs.csv"}[damage]
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed|differ|Source"):
        figures.save_figure_tables(tmp_path / "out", {"summary": summary}, runs,
            targets=targets, selection_directory=root / "hp/selection",
            coverage=coverage, analysis_targets=effective)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("damage", ["missing_run", "config", "eta", "planned_seed"])
def test_shared_input_check_rejects_inconsistent_chapters(saved, damage):
    runs, expected = figures.load_figure_runs(saved)
    runs, expected = copy.deepcopy(runs), expected.copy()
    _, effective = figures.analysis_target_tables(frozen_reference().joined_tables(runs)["summary"])
    targets = effective.set_index("model_code").targ.to_dict()
    first = next(r for r in runs.values() if r[0]["representation"] == "sparse_trained")
    if damage == "missing_run": runs.pop(next(iter(runs)))
    if damage == "config": first[0]["config"]["tau_e"] = 123
    if damage == "eta": first[0]["spec"]["eta"] = 123
    if damage == "planned_seed": expected.loc[0, "seed_index"] = 123
    with pytest.raises(ValueError):
        figures.check_figure_inputs(runs, expected, saved / "hp/selection", targets)
