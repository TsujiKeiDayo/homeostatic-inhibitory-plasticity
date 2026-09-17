"""End-to-end saved-number tests. No experiment, training, inference or UMAP."""
import copy
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notebook_cells import draw_figures, statistics, table_results, paired_results, threshold_results
from figure_fixtures import fixed_experiment, fixed_result
from hnn2.figure_tables import load_figure_runs, validation_points, save_figure_tables, activity_tables, THRESHOLDS
from hnn2.figure_plots import save_basic_figures
from hnn2.result_io import readout_tables


@pytest.fixture(autouse=True)
def forbid_learning(monkeypatch):
    import hnn2.single as single
    import hnn2.workflows as workflows
    import hnn2.mlp as mlp
    import hnn2.umap_embed as umap
    def forbidden(*a, **k):
        pytest.fail("No training/inference/UMAP allowed in figure tests")
    for module, name in ((single, "run_adaptation"), (single, "fit_readout"), (single, "sparse_features"),
                         (workflows, "run_adaptation_batch"), (workflows, "run_representation"),
                         (mlp, "train_mlp_e2e"), (umap, "embed")):
        monkeypatch.setattr(module, name, forbidden)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    # The helper only loads checked-in arrays, authors histories and saves them.
    return fixed_experiment(tmp_path_factory.mktemp("figure_fixed"))


def test_all_models_targets_seeds_to_tables_and_png_pdf(saved, tmp_path):
    runs, expected = load_figure_runs(saved)
    assert len(runs) == len(expected) == 22
    assert set(expected.state) == {"complete"}
    targets = dict.fromkeys(("rec", "ff", "thresh"), .35)
    tables = table_results(runs, expected, saved / "hp/selection", targets)
    assert len(tables["target_evidence"]) == 12
    assert set(tables["target_candidate_evidence"].adoption) == {"not_performed"}
    from hnn2.targets import TIEBREAK_RULE
    for name in ("target_candidate_evidence", "target_diagnostic_candidates"):
        assert set(tables[name].tiebreak) == {TIEBREAK_RULE}
    medians = tables["target_evidence"].groupby(["model_code", "targ"]).final_val_loss.median()
    for row in tables["target_candidate_evidence"].itertuples():
        assert row.median_final_val_loss == medians.loc[(row.model_code, row.candidate_targ)]
    assert tables["endpoint_anchors"].equal.all()
    assert set(tables["threshold_detail"].threshold) == set(THRESHOLDS)
    assert tables["threshold_summary"].expected_total.eq(2).all()
    assert tables["threshold_summary"].missing_or_invalid.eq(0).all()
    assert len(tables["selected_summary"]) == 16  # eight arms, two seeds
    assert set(tables["validation_points"].step) == {2, 4, 6, 8, 9}
    assert set(tables["validation_points"].source) == {"history", "summary"}
    figures = draw_figures(tables, runs, targets)
    try:
        assert {name[:2] for name in figures} == {"G1", "G2", "G3", "G4", "G5", "G6"}
        assert len([name for name in figures if "parameter" in name]) == 6
        output = save_figure_tables(tmp_path / "tables", tables, runs, targets=targets,
                                   selection_directory=saved / "hp/selection")
        files = save_basic_figures(figures, tmp_path / "figures", numeric_source=output)
        assert len(files) == 2 * len(figures)
        assert all(p.stat().st_size > 100 for p in files)
        assert (output / "COMPLETE").is_file()
        with pytest.raises(FileExistsError):
            save_figure_tables(output, tables, runs, targets=targets, selection_directory=saved / "hp/selection")
        with pytest.raises(FileExistsError):
            save_basic_figures(figures, tmp_path / "figures")
        assert save_basic_figures(figures, tmp_path / "off", png=False, pdf=False) == []
        assert not (tmp_path / "off").exists()
    finally:
        for fig in figures.values(): plt.close(fig)


def test_final_only_threshold_reach_and_missing_denominator(saved):
    runs, expected = load_figure_runs(saved)
    runs, expected = copy.deepcopy(runs), expected.copy()
    raw = expected[expected.arm == "raw"]
    first, missing = raw.run.tolist()
    runs[first][2]["summary"].loc[0, "final_val_accuracy"] = .8
    runs[first][2]["summary"].loc[0, "best_val_accuracy"] = .8
    expected.loc[expected.run == missing, ["state", "reason"]] = ["missing", "not_started"]
    detail, summary = threshold_results(runs, expected, dict.fromkeys(("rec", "ff", "thresh"), .35))
    row = summary[(summary.arm == "raw") & (summary.threshold == .8)].iloc[0]
    assert (row.reached, row.total, row.expected_total, row.missing_or_invalid, row.fraction) == (1, 1, 2, 1, 1)
    expected.loc[expected.arm == "raw", "state"] = "missing"
    _, summary = threshold_results(runs, expected, dict.fromkeys(("rec", "ff", "thresh"), .35))
    row = summary[(summary.arm == "raw") & (summary.threshold == .8)].iloc[0]
    assert row.total == 0 and np.isnan(row.fraction)


@pytest.mark.parametrize("key,value", [("tau_e", 123.0), ("n_epochs", 4)])
def test_saved_hp_training_condition_mismatch_rejected(saved, key, value):
    runs, expected = load_figure_runs(saved)
    runs = copy.deepcopy(runs)
    next(iter(runs.values()))[0]["config"][key] = value
    with pytest.raises(ValueError, match="HP and training conditions differ|Training epochs differ"):
        table_results(runs, expected, saved / "hp/selection", dict.fromkeys(("rec", "ff", "thresh"), .35))


@pytest.mark.parametrize("damage", ["nonfinite", "missing_final", "same_step_disagreement", "missing_flag"])
def test_invalid_measurements_not_interpolated(damage):
    t = readout_tables(fixed_result().readout)
    h, s = t["history"], t["summary"]
    if damage == "nonfinite": h.loc[1, "val_accuracy"] = np.nan
    if damage == "missing_final": s.loc[0, "final_val_accuracy"] = np.nan
    if damage == "missing_flag": h = h.drop(columns="val_evaluated")
    if damage == "same_step_disagreement":
        h = h.iloc[:-1].copy()
        s.loc[0, "final_step"] = 8
    with pytest.raises(ValueError): validation_points(h, s)


def test_same_step_deduplicates_and_unobserved_values_do_not_count():
    t = readout_tables(fixed_result().readout)
    h, s = t["history"].iloc[:-1].copy(), t["summary"].copy()
    s.loc[0, ["final_step", "final_val_accuracy", "final_val_loss"]] = [8, h.val_accuracy.iloc[-1], h.val_loss.iloc[-1]]
    h.loc[0, "val_accuracy"] = .99  # Explicitly unmeasured: must not become an observation.
    points = validation_points(h, s)
    assert len(points) == 4
    assert points.source.iloc[-1] == "history+summary"
    assert points.val_accuracy.max() < .99


def test_paired_difference_is_not_difference_of_medians():
    rows = [{"arm": f"rec_{arm}", "model_code": "rec", "targ": .35 if arm == "trained" else np.nan,
             "seed_index": seed, **dict.fromkeys(("final_test_accuracy", "final_test_accuracy_full", "final_val_accuracy"), value)}
            for arm, values in (("trained", [.1, .5, .6]), ("initial", [.0, .1, .9]))
            for seed, value in enumerate(values)]
    _, summary = paired_results(pd.DataFrame(rows))
    assert np.isclose(summary.median_paired_difference.iloc[0], .1)
    assert np.isclose(summary.difference_of_medians.iloc[0], .4)
    with pytest.raises(ValueError): paired_results(pd.DataFrame(rows[:-1]))


def test_descriptive_statistics_keep_missing_count():
    out = statistics(pd.DataFrame({"arm": ["raw"]*3, "v": [1., 3., np.nan]}), ["arm"], ["v"]).iloc[0]
    assert (out.n, out.expected_n, out.missing_or_invalid, out["mean"], out.sd) == (2, 3, 1, 2., 1.)


def test_distribution_axes_exact_zero_and_all_outside_are_distinct():
    result = fixed_result(representation="sparse_initial")
    values = np.array([[0., 0., 3.], [0., 0., 0.]], dtype=np.float32)
    runs = {"fixed": (result.conditions, {"features": {"h_test": values}}, {})}
    metrics, counts = activity_tables(runs)
    pop, life = metrics.set_index("axis").loc["population"], metrics.set_index("axis").loc["lifetime"]
    assert pop.n_values == 2 and life.n_values == 3
    assert np.isclose(pop.strict_zero_fraction, 5/6) and np.isclose(pop.active_fraction, 1/6)
    assert pop.zero_mean_fraction == .5 and np.isclose(life.zero_mean_fraction, 2/3)
    assert counts.groupby("axis")["count"].sum().to_dict() == {"population": 2, "lifetime": 3}
    values[:] = 10
    metrics, _ = activity_tables(runs)
    assert metrics.dropped_mass.eq(1).all() and metrics.S.isna().all()
    assert metrics.strict_zero_fraction.eq(0).all()
    assert metrics.undefined_reason.eq("empty_in_range_histogram").all()
    values[:] = 0
    metrics, _ = activity_tables(runs)
    assert metrics.S.eq(1).all() and metrics.S_prime.eq(0).all()
    assert metrics.strict_zero_fraction.eq(1).all() and metrics.dropped_mass.eq(0).all()


def test_source_change_and_missing_population_refused(saved, tmp_path):
    runs, expected = load_figure_runs(saved)
    targets = dict.fromkeys(("rec", "ff", "thresh"), .35)
    with pytest.raises(ValueError, match="population"):
        table_results(runs, expected.iloc[1:].copy(), saved / "hp/selection", targets)
    first = next(iter(runs))
    runs[first][0]["loaded_source_sha256"]["summary.csv"] = "changed"
    with pytest.raises(ValueError, match="Source changed"):
        save_figure_tables(tmp_path / "out", {}, runs, targets=targets, selection_directory=saved / "hp/selection")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("png,pdf", [(True, False), (False, True)])
def test_independent_figure_format_switches(tmp_path, png, pdf):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 2])
    try:
        files = save_basic_figures({"fixed": fig}, tmp_path / "figures", png=png, pdf=pdf)
        assert [p.suffix for p in files] == ([".png"] if png else [".pdf"])
    finally:
        plt.close(fig)
